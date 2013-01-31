# cython: profile=True
"""
MALT-style dependency parser
"""
cimport cython
import random
from libc.stdlib cimport malloc, free
from pathlib import Path
from collections import defaultdict
import sh
import sys

from _state cimport *
cimport io_parse
import io_parse
from io_parse cimport Sentence
from io_parse cimport Sentences
from io_parse cimport make_sentence
cimport features

from io_parse import LABEL_STRS

import index.hashes
from index.hashes cimport InstanceCounter

cimport svm.cy_svm


cdef int CONTEXT_SIZE = features.CONTEXT_SIZE

VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50
FOLLOW_ERR_PC = 0.90

cdef enum UMove:
    UERR
    USHIFT
    UREDUCE
    ULEFT
    URIGHT
    URIGHT_UNSHIFT
    ULEFT_UNSHIFT
    URIGHT_LOWER
    ULOW_EDGE
    _n_umoves

DEF N_UMOVES = 9
assert N_UMOVES == _n_umoves, "Set N_UMOVES compile var to %d" % _n_umoves

cdef struct LMove:
    UMove umove
    size_t label
    bint is_repair
    size_t id_


cdef lmove_to_str(LMove* m):
    moves = ['E', 'S', 'D', 'L', 'R', 'UR', 'UL', 'RL', 'LE']
    label = LABEL_STRS[m.label]
    if m.umove == USHIFT:
        return 'S'
    elif m.umove == UREDUCE:
        return 'D'
    else:
        return '%s-%s' % (moves[<int>m.umove], label)

cdef transition_to_str(State* s, LMove* parse_move, object tokens):
    tokens = tokens + ['<end>']
    if parse_move.umove == USHIFT:
        return u'%s-->%s' % (tokens[s.i], tokens[s.top])
    elif parse_move.umove == UREDUCE:
        return u'%s/%s' % (tokens[s.top], tokens[s.second])
    elif parse_move.umove == ULOW_EDGE:
        edge = get_right_edge(s, s.top)
        return u'%s<--%s (%s)' % (tokens[s.top], tokens[edge], tokens[s.i])
    elif parse_move.umove == URIGHT_LOWER:
        child = tokens[get_r(s, s.top)]
        parent = tokens[get_r2(s, s.top)]
        top = tokens[s.top]
        return u'%s(%s, %s) ---> %s(%s(%s))' % (top, parent, child, top, parent, child)
    else:
        if parse_move.umove == ULEFT:
            head = s.i
            child = s.top
        elif parse_move.umove == URIGHT_UNSHIFT:
            head = s.second
            child = s.top
        else:
            head = s.top
            child = s.i if s.i < len(tokens) else 0
        return u'%s(%s)' % (tokens[head], tokens[child])


cdef class Parser:
    cdef size_t n_features
    cdef svm.cy_svm.Model guide
    cdef svm.cy_svm.Model l_labeller
    cdef svm.cy_svm.Model r_labeller
    cdef object model_dir
    cdef Sentence* sentence
    cdef int n_preds
    cdef size_t* _context
    cdef size_t* _hashed_feats
    cdef bint* _valid_classes
    cdef TransitionSystem moves
    cdef InstanceCounter inst_counts
    cdef object add_extra
    cdef object label_set
    cdef int feat_thresh
    cdef int n_l
    cdef int n_r

    def __cinit__(self, model_dir, clean=False, C=None, solver_type=None, eps=None,
                  add_extra=True, label_set='MALT', feat_thresh=5,
                  allow_reattach=False, allow_unshift=False, allow_invert=False,
                  allow_move=False, reuse_idx=False, grammar_loc=None):
        model_dir = Path(model_dir)
        if not clean:
            print "Reading settings from config"
            params = dict([line.split() for line in model_dir.join('parser.cfg').open()])
            C = float(params['C'])
            solver_type = int(params['solver_type'])
            eps = float(params['eps'])
            add_extra = True if params['add_extra'] == 'True' else False
            label_set = params['label_set']
            feat_thresh = int(params['feat_thresh'])
            allow_reattach = params['allow_reattach'] == 'True'
            allow_unshift = params['allow_unshift'] == 'True'
            allow_invert = params['allow_invert'] == 'True'
            allow_move = params['allow_move'] == 'True'
            grammar_loc = params['grammar_loc']
            if grammar_loc == 'None':
                grammar_loc = None
            else:
                grammar_loc = Path(str(grammar_loc))

        self.model_dir = self.setup_model_dir(model_dir, clean)
        io_parse.set_labels(label_set)
        self.n_preds = features.make_predicates(add_extra, True)
        self.add_extra = add_extra
        self.label_set = label_set
        self.feat_thresh = feat_thresh
        if solver_type is None:
            solver_type = 6
        if C is None:
            C = 1.0
        if reuse_idx:
            print "Using pre-loaded idx"
        elif clean == True:
            self.new_idx(self.model_dir, self.n_preds)
        else:
            self.load_idx(self.model_dir, self.n_preds)
        self.moves = TransitionSystem(io_parse.LABEL_STRS, allow_reattach=allow_reattach,
                                      allow_unshift=allow_unshift, allow_move=allow_move,
                                      allow_invert=allow_invert, grammar_loc=grammar_loc)
 
        self.guide = svm.cy_svm.Model(self.model_dir.join('model'),
                                      solver_type=solver_type, C=C, eps=eps)
        self.l_labeller = svm.cy_svm.Model(self.model_dir.join('left_label_model'),
                                           solver_type=solver_type)
        self.r_labeller = svm.cy_svm.Model(self.model_dir.join('right_label_model'),
                                           solver_type=solver_type)
        if eps is not None:
            self.guide.eps = eps
        if C is not None:
            self.guide.C = C
        self._context = features.init_context()
        self._hashed_feats = features.init_hashed_features()
        self._valid_classes = <bint*>malloc(N_UMOVES * sizeof(bint))
        self.inst_counts = InstanceCounter()

    def setup_model_dir(self, loc, clean):
        if clean and loc.exists():
            sh.rm('-rf', loc)
        if loc.exists():
            assert loc.is_dir()
        else:
            loc.mkdir()
        sh.git.log(n=1, _out=loc.join('version').open('wb'), _bg=True) 
        return loc

    def train(self, Sentences sents, C=None, solver_type=None, eps=None,
              subparser=None):
        cdef:
            int n_instances
            int i
            Sentence* sent
            State s
        if C is not None:
            self.guide.C = C
        if eps is not None:
            self.guide.eps = eps
        if solver_type is not None:
            self.guide.solver_type = solver_type
        self.write_cfg(self.model_dir.join('parser.cfg'))
       
        # Build the instances without sending them to the learner, so that we
        # can use a frequency threshold on the features.
        index.hashes.set_feat_counting(True)
        index.hashes.set_feat_threshold(self.feat_thresh)
        n_instances = 0
        for i in range(sents.length):
            if sents.s[i].parse.n_moves != 0:
                n_instances += self.follow_moves(&sents.s[i], True)
            else:
                n_instances += self.follow_gold(&sents.s[i], True)
        self.guide.begin_adding_instances(n_instances*3)
        self.l_labeller.begin_adding_instances(n_instances*3)
        self.r_labeller.begin_adding_instances(n_instances*3)
        index.hashes.set_feat_counting(False)
        for i in range(sents.length):
            if sents.s[i].parse.n_moves != 0:
                self.follow_moves(&sents.s[i], False)
            else:
                self.follow_gold(&sents.s[i], False)
        self.guide.train()
        self.l_labeller.train()
        self.r_labeller.train()

    cdef int follow_gold(self, Sentence* sent, bint only_count) except -1:
        cdef LMove* move
        cdef int n_instances = 0
        cdef State s = init_state(sent.length)

        while not s.is_finished:
            features.extract(self._context, self._hashed_feats, sent, &s)
            move = self.moves.static_oracle(&s, sent.pos, sent.parse.labels, sent.parse.heads)
            n_instances += 1
            if not only_count:
                self._add_instance(move.umove, move.label, self.n_preds,
                                   self._hashed_feats, 1)
            self.moves.transition(move, &s)
        return n_instances

    cdef int follow_moves(self, Sentence* sent, bint only_count) except -1:
        cdef LMove* omove
        cdef size_t i
        cdef State s
        cdef int n_instances = 0
        cdef double freq = 0
        s = init_state(sent.length)
        for i in range(sent.parse.n_moves):
            parse_move = &self.moves.lmoves[sent.parse.moves[i]]
            oracle_moves = self.moves.oracle(&s, sent.pos, sent.parse.labels, sent.parse.heads)
            for oid in oracle_moves:
                # TODO: Is this equivalent to old version?
                if self.moves.lmoves[oid].umove == parse_move.umove:
                    oracle_moves = [oid]
                    break
            features.extract(self._context, self._hashed_feats, sent, &s)
            n_moves = float(len(oracle_moves))
            for move_id in oracle_moves:
                omove = &self.moves.lmoves[move_id]
                n_instances += 1
                if not only_count:
                    # TODO: Fix this to refer to the MAX_TRANSITIONS constant
                    freq = self.inst_counts.add(omove.umove, sent.id, 256 * 5,
                                                s.history, not only_count)
                    self._add_instance(omove.umove, omove.label, self.n_preds,
                                       self._hashed_feats, freq / n_moves)
            self.moves.transition(parse_move, &s)
        return n_instances

    cdef int _add_instance(self, UMove umove, size_t label, size_t n_feats,
                           size_t* feats, double freq) except -1:
        if freq > 0:
            if umove == ULEFT:
                self.l_labeller.add_instance(label, 1, n_feats, feats)
            elif umove == URIGHT:
                self.r_labeller.add_instance(label, 1, n_feats, feats)
            self.guide.add_instance(<int>umove, freq, n_feats, feats)

    def add_parses(self, Sentences sents, Sentences gold=None):
        cdef:
            size_t i
        for i in range(sents.length):
            self.parse(&sents.s[i])
        if gold is not None:
            return sents.evaluate(gold)

    cdef int parse(self, Sentence* sent) except -1:
        cdef State s
        cdef LMove *lmove
        cdef size_t n = sent.length
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        while not s.is_finished:
            if s.stack_len == 1:
                lmove = self.moves.S
            else:
                features.extract(self._context, self._hashed_feats, sent, &s)
                self.moves.validate_moves(&s, sent.pos, self._valid_classes)
                lmove = self.predict_move(self.n_preds, self._hashed_feats, self._valid_classes)
            sent.parse.moves[s.t] = lmove.id_
            sent.parse.n_moves += 1
            self.moves.transition(lmove, &s)
        for i in range(1, sent.length):
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]

    cdef LMove* predict_move(self, size_t n, size_t* feats, bint* valid_classes) except NULL:
        cdef UMove umove
        cdef int label
        umove = <UMove>self.guide.predict_from_ints(n, feats, valid_classes)
        if umove == ULEFT:
            label = <int>self.l_labeller.predict_single(n, feats)
        elif umove == URIGHT:
            label = <int>self.r_labeller.predict_single(n, feats)
        else:
            label = 0
        return self.moves.lookup(umove, label)

    def add_gold_moves(self, Sentences sents):
        cdef size_t i
        print "Calculating gold moves"
        for i in range(sents.length):
            self.moves.add_gold_moves(&sents.s[i], sents.strings[i][0])

    def get_best_moves(self, Sentences sents, Sentences gold):
        """Get a list of move taken/oracle move pairs for output"""
        cdef State s
        cdef size_t n
        cdef object best_moves
        cdef LMove* parse_move
        cdef size_t i
        cdef size_t* g_labels
        cdef size_t* g_heads
        best_moves = []
        for i in range(sents.length):
            sent = &sents.s[i]
            g_labels = gold.s[i].parse.labels
            g_heads = gold.s[i].parse.heads
            n = sent.length
            s = init_state(n)
            sent_moves = []
            tokens = sents.strings[i][0]
            while not s.is_finished:
                best_ids = self.moves.oracle(&s, sent.pos, g_labels, g_heads)
                best_id_str = ','.join(["%d" % id_ for id_ in best_ids])
                best_strs = ','.join([lmove_to_str(&self.moves.lmoves[id_])
                                      for id_ in best_ids])
                parse_move = &self.moves.lmoves[sent.parse.moves[s.t]]
                state_str = transition_to_str(&s, parse_move, tokens)
                sent_moves.append((best_id_str, parse_move.id_,
                                  best_strs, lmove_to_str(parse_move), state_str))
                self.moves.transition(parse_move, &s)
            best_moves.append((u' '.join(tokens), sent_moves))
        return best_moves

    def save(self):
        # Save directory set up on init, so just make sure everything saves
        self.guide.save()
        self.l_labeller.save()
        self.r_labeller.save()

    def load(self):
        self.guide.load()
        self.l_labeller.load()
        self.r_labeller.load()

    def __dealloc__(self):
        free(self._context)
        free(self._hashed_feats)

    def new_idx(self, model_dir, size_t n_predicates):
        index.hashes.init_feat_idx(n_predicates, model_dir.join('features'))
        index.hashes.init_word_idx(model_dir.join('words'))
        index.hashes.init_pos_idx(model_dir.join('pos'))

    def load_idx(self, model_dir, size_t n_predicates):
        model_dir = Path(model_dir)
        index.hashes.load_word_idx(model_dir.join('words'))
        index.hashes.load_pos_idx(model_dir.join('pos'))
        index.hashes.load_feat_idx(n_predicates, model_dir.join('features'))
   
    def write_cfg(self, loc):
        with loc.open('w') as cfg:
            cfg.write(u'model_dir\t%s\n' % self.model_dir)
            cfg.write(u'C\t%s\n' % self.guide.C)
            cfg.write(u'eps\t%s\n' % self.guide.eps)
            cfg.write(u'solver_type\t%s\n' % self.guide.solver_type)
            cfg.write(u'add_extra\t%s\n' % self.add_extra)
            cfg.write(u'label_set\t%s\n' % self.label_set)
            cfg.write(u'grammar_loc\t%s\n' % self.moves.grammar_loc)
            cfg.write(u'feat_thresh\t%d\n' % self.feat_thresh)
            cfg.write(u'allow_reattach\t%s\n' % self.moves.allow_reattach)
            cfg.write(u'allow_unshift\t%s\n' % self.moves.allow_unshift)
            cfg.write(u'allow_invert\t%s\n' % self.moves.allow_invert)
            cfg.write(u'allow_move\t%s\n' % self.moves.allow_move)
        

cdef class TransitionSystem:
    cdef LMove *lmoves
    cdef bint allow_reattach
    cdef bint allow_unshift
    cdef bint allow_move
    cdef bint allow_invert
    cdef size_t n_labels
    cdef object py_labels
    cdef size_t[N_UMOVES] offsets
    cdef bint grammar[50][50][50]
    cdef object grammar_loc
    cdef LMove* E
    cdef LMove* S
    cdef LMove* D
    cdef LMove* R
    cdef LMove* L
    cdef LMove* R_U
    cdef LMove* L_U
    cdef LMove* R_L
    cdef LMove* L_E

    cdef int n_lmoves
    cdef int ru_count
    cdef int lu_count
    cdef int le_count

    def __cinit__(self, object labels, allow_reattach=False, allow_unshift=False,
                  allow_move=False, allow_invert=False, grammar_loc=None):
        self.n_labels = len(labels)
        self.set_moves(labels)
        self.py_labels = labels
        self.allow_unshift = allow_unshift
        self.allow_reattach = allow_reattach
        self.allow_move = allow_move
        self.allow_invert = allow_invert
        self.lu_count = 0
        self.ru_count = 0
        if grammar_loc is not None:
            self.read_grammar(grammar_loc)
        self.grammar_loc = grammar_loc


    cdef int read_grammar(self, loc) except -1:
        cdef size_t i, j, k
        GRAMMAR_THRESH = 1000
        lines = [line.split() for line in loc.open().read().strip().split('\n')]
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    self.grammar[i][j][k] = False
        for head, sib, child, freq in lines:
            freq = int(freq)
            head = index.hashes.encode_pos(head)
            sib = index.hashes.encode_pos(sib)
            child = index.hashes.encode_pos(child)
            if  freq >= GRAMMAR_THRESH:
                self.grammar[head][sib][child] = True
            else:
                self.grammar[head][sib][child] = False

    cdef set_moves(self, object labels):
        cdef:
            UMove umove
            size_t n_labels
            size_t n_lmoves
            size_t i
            size_t offset
            size_t label

        n_labels = len(labels)
        n_lmoves = 0
        for i in range(N_UMOVES):
            umove = <UMove>i
            if umove == USHIFT or umove == UREDUCE or umove == UERR:
                n_lmoves += 1
            else:
                n_lmoves += n_labels
        
        self.lmoves = <LMove*>malloc(n_lmoves * sizeof(LMove))
        offset = 0
        for i in range(N_UMOVES):
            umove = <UMove>i
            self.offsets[i] = offset
            if umove == USHIFT or umove == UREDUCE or umove == UERR:
                self.lmoves[offset] = LMove(id_=offset, umove=umove, label=0,
                                            is_repair=False)
                offset += 1
            else:
                if umove == ULEFT or umove == URIGHT:
                    is_repair = False
                else:
                    is_repair = True
                for label in range(n_labels):
                    self.lmoves[offset] = LMove(id_=offset, umove=umove, label=label,
                                                is_repair=is_repair)
                    offset += 1

        self.n_lmoves = offset
        self.E = self.lookup(UERR, 0)
        self.S = self.lookup(USHIFT, 0)
        self.D = self.lookup(UREDUCE, 0)
        self.L = self.lookup(ULEFT, 0)
        self.R = self.lookup(URIGHT, 0)
        self.R_U = self.lookup(URIGHT_UNSHIFT, 0)
        self.L_U = self.lookup(ULEFT_UNSHIFT, 0)
        self.R_L = self.lookup(URIGHT_LOWER, 0)
        self.L_E = self.lookup(ULOW_EDGE, 0)
        move_strs = []
        for i in range(offset):
            move_strs.append(lmove_to_str(&self.lmoves[i]))
        io_parse.set_moves(move_strs)
        
    cdef LMove* lookup(self, UMove umove, size_t label) except NULL:
        cdef LMove* lmove
        cdef size_t idx
        idx = <size_t>self.offsets[<size_t>umove] + label
        assert label < self.n_labels, label
        lmove = &self.lmoves[idx]
        assert lmove.umove == umove, lmove_to_str(lmove) + str(umove) + str(label) 
        assert lmove.label == label
        return lmove

    cdef int transition(self, LMove *lmove, State *s) except -1:
        cdef size_t head, child, new_parent, new_child
        s.history[s.t] = lmove.id_
        s.t += 1 
        if s.t >= 500:
            raise StandardError, s.t
        if lmove.umove == USHIFT:
            push_stack(s)
        elif lmove.umove == UREDUCE:
            pop_stack(s)
        elif lmove.umove == ULEFT:
            child = pop_stack(s)
            if s.heads[child] != 0:
                del_r_child(s, s.heads[child])
            head = s.i
            add_dep(s, head, child, lmove.label)
        elif lmove.umove == URIGHT:
            child = s.i
            head = s.top
            add_dep(s, head, child, lmove.label)
            push_stack(s)
        elif lmove.umove == URIGHT_UNSHIFT:
            self.do_right_unshift(s, lmove.label)
        elif lmove.umove == ULEFT_UNSHIFT:
            self.do_left_unshift(s, lmove.label)
        elif lmove.umove == URIGHT_LOWER:
            self.do_right_lower(s, lmove.label)
        elif lmove.umove == ULOW_EDGE:
            self.do_low_edge(s, lmove.label)
        else:
            raise StandardError(lmove_to_str(lmove))
        if s.i >= (s.n - 1) and s.stack_len == 1:
            s.is_finished = True

    cdef int validate_moves(self, State* s, size_t* tags, bint* valid_moves) except -1:
        cdef:
            bint allow_left, allow_right, allow_right_unshift, allow_left_unshift
            bint allow_low_edge
            size_t id_
            UMove umove
            LMove* prev_move
        for id_ in range(N_UMOVES):
            valid_moves[id_] = False
        if s.t > 0 and s.history[s.t - 1] == self.L_E.id_:
            valid_moves[<int>URIGHT] = True
            return 1
        if s.i < s.n:
            valid_moves[<int>USHIFT] = True
            valid_moves[<int>URIGHT] = True
        if s.stack_len == 1:
            valid_moves[<int>UREDUCE] = False
            valid_moves[<int>ULEFT] = False
            valid_moves[<int>URIGHT] = False
        elif s.heads[s.top] == 0:
            valid_moves[<int>UREDUCE] = s.i == s.n
            valid_moves[<int>ULEFT] = True
        else:
            valid_moves[<int>UREDUCE] = True
            valid_moves[<int>ULEFT] = self.allow_reattach
        valid_moves[<int>URIGHT_UNSHIFT] = self.check_right_unshift(s, NULL)
        valid_moves[<int>ULEFT_UNSHIFT] = self.check_left_unshift(s, NULL)
        valid_moves[<int>URIGHT_LOWER] = self.check_right_lower(s, NULL)
        valid_moves[<int>ULOW_EDGE] = self.check_low_edge(s, NULL)

    cdef object oracle(self, State *s, size_t* tags, size_t* g_labels, size_t* g_heads):
        cdef:
            size_t buff_i, stack_i
        actions = []
        if s.stack_len == 1:
            return [self.S.id_]
        if s.i == s.n:
            return [self.D.id_]
        if g_heads[s.top] == s.i and (s.heads[s.top] == 0 or self.allow_reattach):
            return [self.lookup(ULEFT, g_labels[s.top]).id_]
        if g_heads[s.i] == s.top:
            return [self.lookup(URIGHT, g_labels[s.i]).id_]
        if self.check_right_lower(s, g_heads):
            return [self.lookup(URIGHT_LOWER, g_labels[get_r(s, s.top)]).id_]
        
        s_cost = 0
        d_cost = 0
        l_cost = 0
        r_cost = 0
        rightmost = get_r(s, s.top)
        for buff_i in range(s.i, s.n):
            if g_heads[s.top] == buff_i:
                l_cost += 1
                if self.allow_reattach:
                    d_cost += 1
            # If top's child is in buffer (s, l, k=buff)
            if g_heads[buff_i] == s.top:
                l_cost += 1
                if not (self.allow_move and s.heads[s.top] != s.second):
                    d_cost += 1
            # If word's head is in buffer: (k=buff, l, b)
            if not self.allow_reattach and g_heads[s.i] == buff_i:
                r_cost += 1

        assert s.stack_len >= 1
        assert s.stack[0] == 0
        for stack_i in range(1, s.stack_len):
            # If word's head is in stack (k=stack, l, b)
            if g_heads[s.i] == s.stack[stack_i]:
                s_cost += 1
                r_cost += 1
            # If the word's child is in stack (b, l, k=stack)
            if g_heads[s.stack[stack_i]] == s.i:
                s_cost += 1
                r_cost += 1
            # If word's head is rightmost child of stack item
            if self.allow_move and stack_i != s.top:
                if g_heads[s.i] == get_r(s, stack_i):
                    s_cost += 1
                    r_cost += 1
        if self.allow_move and g_heads[s.i] == get_r(s, s.top):
            s_cost += 1
            d_cost += 1
            if self.allow_reattach or s.heads[s.top] == 0:
                l_cost += 1
        # NB: these being above min_cost is a substantial bug fix
        if s.heads[s.top] == 0:
            d_cost = 10000
        if s.heads[s.top] != 0 and not self.allow_reattach:
            l_cost = 10000
        # Penalise left-arc for clobbering good dependencies
        if self.allow_reattach and s.heads[s.top] == g_heads[s.top]:
            l_cost += 1
        if s_cost == l_cost == d_cost == r_cost:
            return []
        min_cost = min((s_cost, d_cost, l_cost, r_cost))
        if min_cost != 0:
            return []
        actions = []
        if s_cost == min_cost:
            actions.append(self.S.id_)
        if d_cost == min_cost:
            actions.append(self.D.id_)
        if l_cost == min_cost:
            actions.append(self.L.id_)
        if r_cost == min_cost:
            actions.append(self.R.id_)
        return actions

    cdef int check_r_grammar(self, State* s, size_t* tags):
        if self.grammar_loc is None:
            return True
        sib = get_r(s, s.top)
        if sib == 0:
            sib = io_parse.NONE_POS
        else:
            sib = tags[sib] 
        return self.grammar[tags[s.top]][sib][tags[s.i]]

    cdef LMove* static_oracle(self, State *s, size_t* tags, size_t* labels, size_t* heads) except NULL:
        cdef size_t right_edge
        if s.i == s.n:
            return self.D
        elif s.stack_len == 1:
            return self.S
        if self.check_right_lower(s, heads):
            return self.R_L
        if self.allow_move and heads[s.i] > 0 and heads[s.i] == get_r(s, s.top):
            return self.R
        if heads[s.top] == s.i:
            assert s.heads[s.top] <= 1 or self.allow_reattach
            return self.lookup(ULEFT, labels[s.top])

        if heads[s.i] == s.top:
            #if self.allow_move and \
            #  s.heads[s.top] == s.second and \
            #  s.second != 0 and \
            #  labels[s.i] != io_parse.PUNCT_LABEL and \
            #  self.grammar[tags[s.second]][tags[s.top]][tags[s.i]]:
            #    return self.D
            #else:
            return self.lookup(URIGHT, labels[s.i])
        if self.allow_move and s.second > 0 and heads[s.top] == get_r(s, s.second):
            return self.D

        if heads[s.top] > s.i:
            if self.allow_reattach and self.check_r_grammar(s, tags):
                return self.R
            else:
                return self.S
        # For right-lower situations
        if self.allow_move and heads[s.i] < s.i:
            assert self.allow_move
            return self.D
        
        for i in range(1, s.stack_len - 1):
            if heads[s.i] == s.stack[i] or heads[s.stack[i]] == s.i:
                return self.D
        else:
            assert heads[s.i] > s.i, '%d heads %d with %d on stack' % (heads[s.i], s.i, s.top)
            if self.allow_reattach and \
              self.check_r_grammar(s, tags):
                return self.R
            else:
                return self.S

    cdef add_gold_moves(self, Sentence* sent, object py_words):
        cdef State s
        cdef LMove* lmove
        s = init_state(sent.length)
        assert sent.parse.n_moves == 0
        while not s.is_finished: 
            lmove = self.static_oracle(&s, sent.pos, sent.parse.labels, sent.parse.heads)
            sent.parse.moves[s.t] = lmove.id_
            sent.parse.n_moves += 1
            sent.parse.scores[s.t] = 1.0
            self.transition(lmove, &s)

    cdef bint check_right_unshift(self, State *s, size_t* g_heads):
        if not self.allow_unshift:
            return False
        if s.heads[s.top] >= 2:
            return False
        if s.second <= 1:
            return False
        if g_heads is not NULL and g_heads[s.top] != s.second:
            return False
        return True

    cdef int do_right_unshift(self, State *s, size_t label) except -1:
        add_dep(s, s.second, s.top, label)

    cdef bint check_left_unshift(self, State *s, size_t* g_heads):
        if not self.allow_unshift:
            return False
        # This is necessary because allow unshift when top is attached to second
        # could lead to cycles between this and left-invert
        if s.heads[s.top] >= 2:
            return False
        if s.second <= 1:
            return False
        # TODO: Experiment with this
        if s.heads[s.second] >= 2 and not self.allow_reattach:
            return False
        if g_heads is not NULL and g_heads[s.second] != s.top:
            return False
        return True

    cdef int do_left_unshift(self, State *s, size_t label) except -1:
        if s.heads[s.second] >= 2:
            del_r_child(s, s.heads[s.second])
        add_dep(s, s.top, s.second, label)
        s.stack_len -= 1
        s.stack[s.stack_len - 1] = s.top
        if s.stack_len >= 2:
            s.second = s.stack[s.stack_len - 2]
        else:
            s.second = 1

    cdef bint check_low_edge(self, State *s, size_t* g_heads):
        cdef size_t node, leftmost
        if not self.allow_invert:
            return False
        if s.top < 2:
            return False
        if s.i == s.n:
            return False
        node = get_r(s, s.top)
        if node == 0:
            return False
        assert s.r_valencies[s.top] != 0
        if g_heads == NULL:
            return True
        while s.r_valencies[node] != 0:
            assert node != 0
            node = get_r(s, node)
        if g_heads[s.i] == node:
            return True
        return False

    cdef int do_low_edge(self, State *s, size_t label) except -1:
        edge = s.top
        while s.r_valencies[edge] != 0:
            edge = get_r(s, edge)
            assert edge != 0
            # Restore all nodes in between the new parent and top to the stack
            s.second = s.top
            s.top = edge
            s.stack[s.stack_len] = edge
            s.stack_len += 1

    cdef int check_right_lower(self, State *s, size_t* g_heads) except -1:
        if s.r_valencies[s.top] < 2:
            return False
        if not self.allow_move:
            return False
        if s.top < 1:
            return False
        if g_heads == NULL:
            return True
        r = get_r(s, s.top)
        r2 = get_r2(s, s.top)
        if g_heads[r] != r2:
            return False
        return True

    cdef int do_right_lower(self, State *s, size_t label) except -1:
        cdef size_t r, r2
        assert s.r_valencies[s.top] >= 2
        r = get_r(s, s.top)
        r_label = s.labels[r]
        r2 = get_r2(s, s.top)
        del_r_child(s, s.top)
        add_dep(s, r2, r, label)
        s.second = r2
        s.stack[s.stack_len] = r2
        s.stack_len += 1
        s.top = r
        s.stack[s.stack_len] = r
        s.stack_len += 1
