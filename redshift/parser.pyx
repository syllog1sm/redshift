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

from svm.cy_svm cimport Model, LibLinear, Perceptron

random.seed(0)


cdef int CONTEXT_SIZE = features.CONTEXT_SIZE

VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50
REPAIR_ONLY = True
cdef double FOLLOW_ERR_PC = 0.90


cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    LOWER
    _n_moves

DEF N_MOVES = 6
assert N_MOVES == _n_moves, "Set N_MOVES compile var to %d" % _n_moves


DEF USE_COLOURS = True

def red(string):
    if USE_COLOURS:
        return u'\033[91m%s\033[0m' % string
    else:
        return string


cdef lmove_to_str(move, label):
    moves = ['E', 'S', 'D', 'L', 'R', 'W']
    label = LABEL_STRS[label]
    if move == SHIFT:
        return 'S'
    elif move == REDUCE:
        return 'D'
    else:
        return '%s' % (moves[move])


cdef class Parser:
    cdef size_t n_features
    cdef Model guide
    cdef object model_dir
    cdef Sentence* sentence
    cdef int n_preds
    cdef size_t* _context
    cdef size_t* _hashed_feats
    cdef TransitionSystem moves
    cdef InstanceCounter inst_counts
    cdef object add_extra
    cdef object label_set
    cdef object train_alg
    cdef int feat_thresh

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  shiftless=False, repair_only=False, n_iters=15,
                  add_extra=True, label_set='MALT', feat_thresh=5,
                  allow_reattach=False, allow_lower=False,
                  reuse_idx=False, grammar_loc=None):
        model_dir = Path(model_dir)
        if not clean:
            params = dict([line.split() for line in model_dir.join('parser.cfg').open()])
            C = float(params['C'])
            train_alg = params['train_alg']
            eps = float(params['eps'])
            add_extra = True if params['add_extra'] == 'True' else False
            label_set = params['label_set']
            feat_thresh = int(params['feat_thresh'])
            allow_reattach = params['allow_reattach'] == 'True'
            allow_lower = params['allow_lower'] == 'True'
            grammar_loc = params['grammar_loc']
            shiftless = params['shiftless'] == 'True'
            repair_only = params['repair_only'] == 'True'
            if grammar_loc == 'None':
                grammar_loc = None
            else:
                grammar_loc = Path(str(grammar_loc))
        if allow_reattach:
            print 'Reattach'
        if allow_lower:
            print 'Lower'
        if shiftless:
            print 'Shiftless'
        if repair_only:
            print 'Repair only'
        self.model_dir = self.setup_model_dir(model_dir, clean)
        io_parse.set_labels(label_set)
        self.n_preds = features.make_predicates(add_extra, True)
        self.add_extra = add_extra
        self.label_set = label_set
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        if clean == True:
            self.new_idx(self.model_dir, self.n_preds)
        else:
            self.load_idx(self.model_dir, self.n_preds)
        if shiftless:
            assert not repair_only
        self.moves = TransitionSystem(io_parse.LABEL_STRS, allow_reattach=allow_reattach,
                                      allow_lower=allow_lower, grammar_loc=grammar_loc,
                                      shiftless=shiftless, repair_only=repair_only)
        guide_loc = self.model_dir.join('model')
        n_labels = len(io_parse.LABEL_STRS)
        self.guide = Perceptron(self.moves.n_paired, guide_loc)
        self._context = features.init_context()
        self._hashed_feats = features.init_hashed_features()
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

    def train(self, Sentences sents, C=None, eps=None, n_iter=15):
        self.write_cfg(self.model_dir.join('parser.cfg'))
        index.hashes.set_feat_counting(True)
        index.hashes.set_feat_threshold(self.feat_thresh)
        indices = range(sents.length)
        # First iteration of static is used to collect feature frequencies, so
        # we need an extra iteration
        for n in range(n_iter + (1 if self.train_alg == 'static' else 0)):
            random.shuffle(indices)
            for i in indices:
                if self.train_alg == 'online':
                    self.online_train_one(n, &sents.s[i])
                else:
                    self.static_train_one(n, &sents.s[i])
            if n == 0:
                index.hashes.set_feat_counting(False)
            if n > 0 or self.train_alg == "online":
                acc = (float(self.guide.n_corr) / self.guide.total) * 100
                print "Iter #%d %d/%d=%.2f" % (n, self.guide.n_corr, self.guide.total, acc)
                self.guide.n_corr = 0
                self.guide.total = 0
        self.guide.train()

    cdef int static_train_one(self, size_t iter_num, Sentence* sent) except -1:
        cdef size_t move
        cdef size_t label

        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads
        cdef bint* valid
        cdef size_t* tags = sent.pos
        cdef size_t* feats = self._hashed_feats
        cdef size_t* context = self._context
        cdef int n_feats = self.n_preds

        cdef State s = init_state(sent.length)
        cdef int n_instances = 0
        while not s.is_finished:
            features.extract(self._context, self._hashed_feats, sent, &s)
            # Determine which moves are zero-cost and meet pre-conditions
            valid = self.moves.check_costs(&s, g_labels, g_heads)
            # Translates the result of that into the "static oracle" move,
            # i.e. it decides which single move to take.
            move = self.moves.break_tie(&s, g_labels, g_heads, tags, valid)
            if move == LEFT:
                label = g_labels[s.top]
            elif move == RIGHT:
                label = g_labels[s.i]
            else:
                label = 0
            paired = self.moves.pair_label_move(label, move)
            if iter_num != 0:
                self.guide.add_instance(paired, 1.0, n_feats, feats)
            self.moves.transition(move, label, &s)
            n_instances += 1
        return n_instances

    cdef int online_train_one(self, int iter_num, Sentence* sent) except -1:
        cdef size_t move, label, gold_move, gold_label, pred_move, pred_label
        cdef bint* valid
        
        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads
        cdef size_t* tags = sent.pos

        cdef size_t* context = self._context
        cdef size_t* feats = self._hashed_feats
        cdef size_t n_feats = self.n_preds
        cdef Model labeller

        cdef State s = init_state(sent.length)
        cdef int n_instances = 0
        while not s.is_finished:
            features.extract(self._context, self._hashed_feats, sent, &s)
            # Determine which moves are zero-cost and meet pre-conditions
            valid = self.moves.check_preconditions(&s)
            pred_paired = self.guide.predict_from_ints(n_feats, feats, valid)
            valid = self.moves.check_costs(&s, g_labels, g_heads)
            gold_paired = self.guide.predict_from_ints(n_feats, feats, valid)
            self.guide.update(pred_paired, gold_paired, n_feats, feats)
            if valid[pred_paired]:
                gold_paired = pred_paired
            else:
                gold_paired = self.guide.predict_from_ints(n_feats, feats, valid)
            if iter_num >= 2 and random.random() < FOLLOW_ERR_PC:
                self.moves.unpair_label_move(pred_paired, &label, &move)
            else:
                self.moves.unpair_label_move(gold_paired, &label, &move)
            self.moves.transition(move, label, &s)

    def add_parses(self, Sentences sents, Sentences gold=None):
        cdef:
            size_t i
        for i in range(sents.length):
            self.parse(&sents.s[i])
        if gold is not None:
            return sents.evaluate(gold)

    cdef int parse(self, Sentence* sent) except -1:
        cdef State s
        cdef size_t move, label
        cdef size_t n_preds = self.n_preds
        cdef size_t* context = self._context
        cdef size_t* feats = self._hashed_feats
        cdef bint* valid
        cdef size_t n = sent.length
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        while not s.is_finished:
            #if s.top == 0:
            #    move = SHIFT
            #    label = 0
            #else:
            valid = self.moves.check_preconditions(&s)
            features.extract(context, feats, sent, &s)
            paired = self.guide.predict_from_ints(n_preds, feats, valid)
            self.moves.unpair_label_move(paired, &label, &move)
            sent.parse.moves[s.t] = paired
            sent.parse.n_moves += 1
            self.moves.transition(move, label, &s)
        for i in range(1, sent.length):
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]

    def save(self):
        self.guide.save(self.model_dir.join('model'))

    def load(self):
        self.guide.load(self.model_dir.join('model'))

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
            cfg.write(u'train_alg\t%s\n' % self.train_alg)
            cfg.write(u'add_extra\t%s\n' % self.add_extra)
            cfg.write(u'label_set\t%s\n' % self.label_set)
            cfg.write(u'grammar_loc\t%s\n' % self.moves.grammar_loc)
            cfg.write(u'feat_thresh\t%d\n' % self.feat_thresh)
            cfg.write(u'allow_reattach\t%s\n' % self.moves.allow_reattach)
            cfg.write(u'allow_lower\t%s\n' % self.moves.allow_lower)
            cfg.write(u'shiftless\t%s\n' % self.moves.shiftless)
            cfg.write(u'repair_only\t%s\n' % self.moves.repair_only)
        
    def get_best_moves(self, Sentences sents, Sentences gold):
        """Get a list of move taken/oracle move pairs for output"""
        cdef State s
        cdef size_t n
        cdef size_t move = 0
        cdef size_t label = 0
        cdef object best_moves
        cdef size_t i
        cdef size_t* g_labels
        cdef size_t* g_heads
        cdef size_t paired, parse_paired
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
                valid = self.moves.check_costs(&s, g_labels, g_heads)
                best_strs = []
                best_ids = set()
                for paired in range(self.moves.n_paired):
                    if valid[paired]:
                        self.moves.unpair_label_move(paired, &label, &move)
                        if move not in best_ids:
                            best_strs.append(lmove_to_str(move, label))
                        best_ids.add(move)
                best_strs = ','.join(best_strs)
                best_id_str = ','.join(map(str, sorted(best_ids)))
                parse_paired = sent.parse.moves[s.t]
                self.moves.unpair_label_move(parse_paired, &label, &move)
                state_str = transition_to_str(&s, move, label, tokens)
                parse_move_str = lmove_to_str(move, label)
                if move not in best_ids:
                    parse_move_str = red(parse_move_str)
                sent_moves.append((best_id_str, int(move),
                                  best_strs, parse_move_str,
                                  state_str))
                self.moves.transition(move, label, &s)
            best_moves.append((u' '.join(tokens), sent_moves))
        return best_moves


cdef class TransitionSystem:
    cdef bint allow_reattach
    cdef bint allow_lower
    cdef bint shiftless
    cdef bint repair_only
    cdef size_t n_labels
    cdef object py_labels
    cdef size_t[N_MOVES] offsets
    cdef int grammar[50][50][50]
    cdef int default_labels[50][50][50]
    cdef object grammar_loc
    cdef bint* _move_validity
    cdef bint* _pair_validity
    cdef size_t n_paired
    cdef size_t s_id
    cdef size_t d_id
    cdef size_t l_start
    cdef size_t l_end
    cdef size_t r_start
    cdef size_t r_end
    cdef size_t w_start
    cdef size_t w_end

    cdef int n_lmoves

    def __cinit__(self, object labels, allow_reattach=False,
                  allow_lower=False, grammar_loc=None,
                  shiftless=False, repair_only=False):
        self.n_labels = len(labels)
        self.py_labels = labels
        self.allow_reattach = allow_reattach
        self.allow_lower = allow_lower
        self.shiftless = shiftless
        self.repair_only = repair_only
        if grammar_loc is not None:
            self.read_grammar(grammar_loc)
        self.grammar_loc = grammar_loc
        self._move_validity = <bint*>malloc(N_MOVES * sizeof(bint))
        self._pair_validity = <bint*>malloc(N_MOVES * self.n_labels * sizeof(bint))
        self.n_paired = N_MOVES * self.n_labels
        self.s_id = SHIFT * self.n_labels
        self.d_id = REDUCE * self.n_labels
        self.l_start = LEFT * self.n_labels
        self.l_end = (LEFT + 1) * self.n_labels
        self.r_start = RIGHT * self.n_labels
        self.r_end = (RIGHT + 1) * self.n_labels
        self.w_start = LOWER * self.n_labels
        self.w_end = (LOWER + 1) * self.n_labels

    cdef size_t pair_label_move(self, size_t label, size_t move):
        return move * self.n_labels + label

    cdef int unpair_label_move(self, size_t paired, size_t* label, size_t* move):
        move[0] = paired / self.n_labels
        label[0] = paired % self.n_labels

    cdef int read_grammar(self, loc) except -1:
        cdef size_t i, j, k
        lines = [line.split() for line in loc.open().read().strip().split('\n')]
        for i in range(50):
            for j in range(50):
                for k in range(50):
                    self.grammar[i][j][k] = 0
        for head, sib, child, freq, label in lines:
            freq = int(freq)
            head = index.hashes.encode_pos(head)
            sib = index.hashes.encode_pos(sib)
            child = index.hashes.encode_pos(child)
            self.default_labels[head][sib][child] = io_parse.STR_TO_LABEL.get(label, 0)
            self.grammar[head][sib][child] = freq

    cdef int transition(self, size_t move, size_t label, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc
        s.history[s.t] = self.pair_label_move(label, move)
        s.t += 1 
        if move == SHIFT:
            assert not self.shiftless
            push_stack(s)
        elif move == REDUCE:
            pop_stack(s)
        elif move == LEFT:
            child = pop_stack(s)
            if s.heads[child] != 0:
                del_r_child(s, s.heads[child])
            head = s.i
            add_dep(s, head, child, label)
        elif move == RIGHT:
            child = s.i
            head = s.top
            add_dep(s, head, child, label)
            push_stack(s)
        elif move == LOWER:
            assert s.r_valencies[s.top] >= 2
            gc = get_r(s, s.top)
            c = get_r2(s, s.top)
            del_r_child(s, s.top)
            add_dep(s, c, gc, label)
            s.second = s.top
            s.top = c
            s.stack[s.stack_len] = c
            s.stack_len += 1
        else:
            raise StandardError(lmove_to_str(move, label))
        if s.i >= (s.n - 1) and s.stack_len == 1:
            s.is_finished = True

    cdef bint* check_preconditions(self, State* s) except NULL:
        cdef size_t i, l_id, r_id, w_id
        cdef bint* unpaired = self._move_validity
        # Load pre-conditions that don't refer to gold heads
        unpaired[ERR] = False
        unpaired[SHIFT] = s.i < s.n and not self.shiftless
        unpaired[RIGHT] = s.i < s.n and (s.top != 0 or self.shiftless)
        unpaired[REDUCE] = s.top != 0 and s.heads[s.top] != 0
        unpaired[LEFT] = s.top != 0 and (s.heads[s.top] == 0 or self.allow_reattach)
        unpaired[LOWER] = self.allow_lower and s.r_valencies[s.top] >= 2
        cdef bint* paired = self._pair_validity
        for i in range(self.n_paired):
            paired[i] = False
        paired[self.s_id] = unpaired[SHIFT]
        paired[self.d_id] = unpaired[REDUCE]
        for l_id in range(self.l_start, self.l_end):
            paired[l_id] = unpaired[LEFT]
        for r_id in range(self.r_start, self.r_end):
            paired[r_id] = unpaired[RIGHT]
        for w_id in range(self.w_start, self.w_end):
            paired[w_id] = unpaired[LOWER]
        return paired
   
    cdef bint* check_costs(self, State* s, size_t* labels, size_t* heads) except NULL:
        cdef size_t l_id, r_id, w_id
        cdef bint* valid_moves = self._move_validity
        paired_validity = self.check_preconditions(s)
        valid_moves[SHIFT] = paired_validity[self.s_id] and self.s_cost(s, heads)
        valid_moves[REDUCE] = paired_validity[self.d_id] and self.d_cost(s, heads)
        valid_moves[LEFT] = paired_validity[self.l_start] and self.l_cost(s, heads)
        valid_moves[RIGHT] = paired_validity[self.r_start] and self.r_cost(s, heads)
        valid_moves[LOWER] = paired_validity[self.w_start] and self.w_cost(s, heads)
        paired_validity[self.s_id] = valid_moves[SHIFT]
        paired_validity[self.d_id] = valid_moves[REDUCE]
        if valid_moves[LEFT] and heads[s.top] == s.i:
            for l_id in range(self.l_start, self.l_end):
                paired_validity[l_id] = False
            paired_validity[self.pair_label_move(labels[s.top], LEFT)] = True
        else:
            for l_id in range(self.l_start, self.l_end):
                paired_validity[l_id] = valid_moves[LEFT]
        if valid_moves[RIGHT] and heads[s.i] == s.top:
            for r_id in range(self.r_start, self.r_end):
                paired_validity[r_id] = False
            paired_validity[self.pair_label_move(labels[s.i], RIGHT)] = True
        else:
            for r_id in range(self.r_start, self.r_end):
                paired_validity[r_id] = valid_moves[RIGHT]
        if valid_moves[LOWER] and heads[get_r(s, s.top)] == get_r2(s, s.top):
            for w_id in range(self.w_start, self.w_end):
                paired_validity[w_id] = False
            paired_validity[self.pair_label_move(labels[get_r(s, s.top)], LOWER)] = True
        else:
            for w_id in range(self.w_start, self.w_end):
                paired_validity[w_id] = valid_moves[LOWER]
        return paired_validity

    cdef bint* _fill_paired(self, bint* unpaired) except NULL:
        cdef size_t i, l_id, r_id, w_id


    cdef int break_tie(self, State* s, size_t* labels, size_t* heads,
                       size_t* tags, bint* valid_moves) except -1:
        cdef size_t w_id, l_id
        self.check_costs(s, labels, heads)
        if self._move_validity[RIGHT] and heads[s.i] == s.top:
            return RIGHT
        elif self._move_validity[LEFT] and heads[s.top] == s.i:
            return LEFT
        elif self._move_validity[REDUCE]:
            return REDUCE
        elif self._move_validity[SHIFT]:
            return SHIFT
        elif self._move_validity[RIGHT]:
            return RIGHT
        elif self._move_validity[LEFT]:
            return LEFT
        else:
            raise StandardError

    cdef bint s_cost(self, State *s, size_t* g_heads):
        cdef size_t i, stack_i
        if has_child_in_stack(s, s.i, g_heads):
            return False
        if has_head_in_stack(s, s.i, g_heads):
            return False
        if self.allow_lower:
            for i in range(1, s.stack_len):
                stack_i = s.stack[i]
                if get_r(s, stack_i) != 0 and g_heads[s.i] == get_r(s, stack_i):
                    return False
            if s.r_valencies[s.top] >= 2 and self.w_cost(s, g_heads) \
              and g_heads[s.i] == get_r2(s, s.top):
                return False
        return True

    cdef bint r_cost(self, State *s, size_t* g_heads):
        cdef size_t i, buff_i, stack_i
        if g_heads[s.i] == s.top:
            return True
        if has_head_in_buffer(s, s.i, g_heads) and (not self.allow_reattach or self.repair_only):
            return False
        if has_child_in_stack(s, s.i, g_heads):
            return False
        if has_head_in_stack(s, s.i, g_heads):
            return False
        if self.allow_lower:
            for i in range(1, s.stack_len - 1):
                stack_i = s.stack[i]
                if get_r(s, stack_i) != 0 and g_heads[s.i] == get_r(s, stack_i):
                    return False
            # This is a heuristic, because we could theoretically steal away the
            # bad dependency. But penalise it anyway
            if s.r_valencies[s.top] >= 2 and self.w_cost(s, g_heads):
                return False
        return True

    cdef bint d_cost(self, State *s, size_t* g_heads):
        if has_child_in_buffer(s, s.top, g_heads):
            if not self.allow_lower:
                return False
            elif s.second == 0 or s.heads[s.top] != s.second:
                return False
        if self.allow_reattach and has_head_in_buffer(s, s.top, g_heads):
            return False
        if self.allow_lower and get_r(s, s.top) != 0:
            for buff_i in range(s.i, s.n):
                if g_heads[buff_i] == get_r(s, s.top):
                    return False
            if s.r_valencies[s.top] >= 2 and self.w_cost(s, g_heads):
                return False
        return True

    cdef bint l_cost(self, State *s, size_t* g_heads):
        cdef size_t buff_i
        if g_heads[s.top] == s.i:
            return True
        if has_head_in_buffer(s, s.top, g_heads):
            return False
        if has_child_in_buffer(s, s.top, g_heads):
            return False
        if self.allow_reattach and g_heads[s.top] == s.heads[s.top]:
            return False
        if self.allow_lower:
            for buff_i in range(s.i, s.n):
                if g_heads[buff_i] == get_r(s, s.top):
                    return False
            if s.r_valencies[s.top] >= 2 and self.w_cost(s, g_heads):
                return False
        return True

    cdef bint w_cost(self, State *s, size_t* g_heads):
        return g_heads[get_r(s, s.top)] == get_r2(s, s.top)

    """
    cdef object oracle(self, State* s,  size_t* labels, size_t* heads, size_t* tags,
                       parse_move, size_t parse_label, bint* valid_moves):
        self.validate_moves(s, heads, valid_moves)
        if heads[s.i] == s.top:
            return [(RIGHT, labels[s.i])]
        if heads[s.top] == s.i:
            return [(LEFT, labels[s.top])]
        if valid_moves[parse_move]:
            assert parse_move != ERR
            label = self.get_label(s, tags, parse_move, parse_label, labels, heads)
            return [(parse_move, label)]
        if parse_move == ERR:
            move = self.break_tie(s, labels, heads, tags, valid_moves)
            label = self.get_label(s, tags, move, 0, labels, heads)
            return [(move, label)]
        # If we reduce incorrectly, don't confuse the decision boundary by supplying
        # right or left
        elif valid_moves[SHIFT] and parse_move == REDUCE:
            return [(SHIFT, 0)]
        omoves = []
        for move in range(1, N_MOVES):
            if valid_moves[move]:
                label = self.get_label(s, tags, move, 0, labels, heads)
                omoves.append((move, label))
        return omoves

    cdef int get_label(self, State* s, size_t* tags, size_t move, size_t parse_label,
                       size_t* g_labels, size_t* g_heads) except -1:
        if move == SHIFT:
            return 0
        if move == REDUCE:
            return 0
        if move == LEFT:
            if g_heads[s.top] == s.i:
                return g_labels[s.top]
            else:
                return parse_label
        elif move == LOWER:
            if g_heads[get_r(s, s.top)] == get_r2(s, s.top):
                return g_labels[get_r(s, s.top)]
            else:
                return 0
        elif move == RIGHT:
            if g_heads[s.i] == s.top:
                return g_labels[s.i]
            elif parse_label != 0:
                return parse_label
            elif move == RIGHT:
                sib = get_r(s, s.top)
                sib_pos = tags[sib] if sib != 0 else index.hashes.encode_pos('NONE')
                return self.default_labels[tags[s.top]][sib_pos][tags[s.i]]
            else:
                return 0
        return -1


    cdef int follow_moves(self, Sentence* sent, bint only_count, object py_words) except -1:
        cdef size_t i = 0
        cdef double freq = 0
        cdef State s = init_state(sent.length)
        cdef int n_instances = 0
        while not s.is_finished:
            p_move = sent.parse.moves[i]
            p_label = sent.parse.move_labels[i]
            i += 1
            o_moves = self.moves.oracle(&s, sent.parse.labels, sent.parse.heads,
                                        sent.pos, p_move, p_label,
                                        self._valid_classes)
            features.extract(self._context, self._hashed_feats, sent, &s)
            self._add_instance(sent.id, s.history, o_moves,
                               self.n_preds, self._hashed_feats, only_count)
            n_instances += len(o_moves)
            #print py_words[s.top],
            if p_move == ERR:
                assert len(o_moves) == 1
                self.moves.transition(o_moves[0][0], o_moves[0][1], &s)
            else:
                self.moves.transition(p_move, p_label, &s)
            #print py_words[s.top]
        return n_instances

    cdef int _add_instance(self, size_t sent_id, size_t* history, object moves,
                           size_t n_feats, size_t* feats, bint only_count) except -1:
        n_moves = len(moves)
        for move, label in moves:
            if self.guide.solver_type != PERCEPTRON_SOLVER:
                freq = self.inst_counts.add(move, sent_id, history, not only_count)
            else:
                freq = 1
            assert move != ERR
            if freq > 0 and not only_count:
                assert move != ERR
                self.guide.add_instance(move, float(freq) / n_moves, n_feats, feats)
                if move == LEFT:
                    self.l_labeller.add_instance(label, 1, n_feats, feats)
                elif move == RIGHT:
                    self.r_labeller.add_instance(label, 1, n_feats, feats)


    def train_svm(self, Sentences sents, C=None, eps=None):
        cdef:
            int i
            Sentence* sent
            State s

        if C is not None:
            self.guide.C = C
        if eps is not None:
            self.guide.eps = eps
        # Build the instances without sending them to the learner, so that we
        # can use a frequency threshold on the features.
        index.hashes.set_feat_counting(True)
        index.hashes.set_feat_threshold(self.feat_thresh)
        cdef int n_instances = 0
        for i in range(sents.length):
            n_instances += self.follow_moves(&sents.s[i], True, sents.strings[i][0])
        index.hashes.set_feat_counting(False)
        self.guide.begin_adding_instances(n_instances)
        self.l_labeller.begin_adding_instances(n_instances)
        self.r_labeller.begin_adding_instances(n_instances)
        for i in range(sents.length):
            self.follow_moves(&sents.s[i], False, sents.strings[i][0])
        self.guide.train()
        self.l_labeller.train()
        self.r_labeller.train()


        """





cdef transition_to_str(State* s, size_t move, label, object tokens):
    tokens = tokens + ['<end>']
    if move == SHIFT:
        return u'%s-->%s' % (tokens[s.i], tokens[s.top])
    elif move == REDUCE:
        return u'%s/%s' % (tokens[s.top], tokens[s.second])
    elif move == LOWER:
        child = tokens[get_r(s, s.top)]
        parent = tokens[get_r2(s, s.top)]
        top = tokens[s.top]
        return u'%s(%s, %s) ---> %s(%s(%s))' % (top, parent, child, top, parent, child)
    else:
        if move == LEFT:
            head = s.i
            child = s.top
        else:
            head = s.top
            child = s.i if s.i < len(tokens) else 0
        return u'%s(%s)' % (tokens[head], tokens[child])
