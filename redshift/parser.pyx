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

from io_parse import LABEL_STRS, STR_TO_LABEL

import index.hashes
from index.hashes cimport InstanceCounter

from svm.cy_svm cimport Model, LibLinear, Perceptron

from libc.stdint cimport uint64_t, int64_t

cdef int CONTEXT_SIZE = features.CONTEXT_SIZE

VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50
cdef double FOLLOW_ERR_PC = 0.90


DEBUG = False 
def set_debug(val):
    global DEBUG
    DEBUG = val

cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    LOWER
    INVERT
    _n_moves

DEF N_MOVES = 7
assert N_MOVES == _n_moves, "Set N_MOVES compile var to %d" % _n_moves


DEF USE_COLOURS = True

def red(string):
    if USE_COLOURS:
        return u'\033[91m%s\033[0m' % string
    else:
        return string


cdef lmove_to_str(move, label):
    moves = ['E', 'S', 'D', 'L', 'R', 'W', 'V']
    label = LABEL_STRS[label]
    if move == SHIFT:
        return 'S'
    elif move == REDUCE:
        return 'D'
    else:
        return '%s-%s' % (moves[move], label)


def _parse_labels_str(labels_str):
    return [STR_TO_LABEL[l] for l in labels_str.split(',')]


cdef class Parser:
    cdef size_t n_features
    cdef Model guide
    cdef object model_dir
    cdef Sentence* sentence
    cdef int n_preds
    cdef size_t* _context
    cdef uint64_t* _hashed_feats
    cdef TransitionSystem moves
    cdef InstanceCounter inst_counts
    cdef object add_extra
    cdef object label_set
    cdef object train_alg
    cdef int feat_thresh

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  shiftless=False, repair_only=False,
                  add_extra=True, label_set='MALT', feat_thresh=5,
                  allow_reattach=False, allow_lower=False,
                  allow_invert=False, reuse_idx=False):
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
            allow_invert = params['allow_invert'] == 'True'
            shiftless = params['shiftless'] == 'True'
            repair_only = params['repair_only'] == 'True'
            l_labels = params['left_labels']
            r_labels = params['right_labels']
        if allow_reattach:
            print 'Reattach'
        if allow_lower:
            print 'Lower'
        if shiftless:
            print 'Shiftless'
        if repair_only:
            print 'Repair only'
        if allow_invert:
            print "Invert"
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
                                      allow_lower=allow_lower, allow_invert=allow_invert,
                                      shiftless=shiftless, repair_only=repair_only)
        if not clean:
            self.moves.set_labels(_parse_labels_str(l_labels), _parse_labels_str(r_labels))
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

    def train(self, Sentences sents, C=None, eps=None, n_iter=15, held_out=None):
        cdef size_t i, j, n
        cdef Sentence* sent
        cdef Sentences held_out_gold
        cdef Sentences held_out_parse
        # Count classes and labels
        seen_l_labels = set([])
        seen_r_labels = set([])
        seen_classes = set([self.moves.d_id])
        if not self.moves.shiftless:
            seen_classes.add(self.moves.s_id)
        if self.moves.allow_lower:
            seen_classes.add(self.moves.w_start)
        for i in range(sents.length):
            sent = &sents.s[i]
            for j in range(1, sent.length - 1):
                label = sent.parse.labels[j]
                if sent.parse.heads[j] > j:
                    seen_classes.add(self.moves.pair_label_move(label, LEFT))
                    seen_l_labels.add(label)
                else:
                    seen_r_labels.add(label)
                    seen_classes.add(self.moves.pair_label_move(label, RIGHT))
        move_classes = self.moves.set_labels(seen_l_labels, seen_r_labels)
        print "%d vs %d classes seen" % (len(seen_classes), len(move_classes))
        self.guide.set_classes(move_classes)
        self.write_cfg(self.model_dir.join('parser.cfg'))
        indices = range(sents.length)
        for n in range(n_iter):
            random.shuffle(indices)
            for sent_id, i in enumerate(indices):
                if self.train_alg == 'online':
                    self.online_train_one(n, &sents.s[i], sents.strings[i][0])
                else:
                    self.static_train_one(n, &sents.s[i])
            move_acc = (float(self.guide.n_corr) / self.guide.total) * 100
            print "#%d: Moves %d/%d=%.2f" % (n, self.guide.n_corr,
                                             self.guide.total, move_acc)
            self.guide.prune(5)
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
        cdef uint64_t* feats = self._hashed_feats
        cdef size_t* context = self._context
        cdef int n_feats = self.n_preds

        cdef State s = init_state(sent.length)
        cdef int n_instances = 0
        cdef size_t p_move
        cdef size_t p_label
        while not s.is_finished:
            features.extract(self._context, self._hashed_feats, sent, &s)
            # Determine which moves are zero-cost and meet pre-conditions
            valid = self.moves.check_costs(&s, g_labels, g_heads)
            # Translates the result of that into the "static oracle" move,
            # i.e. it decides which single move to take.
            move = self.moves.break_tie(&s, g_labels, g_heads, tags, valid)
            if move == LEFT:
                label = g_labels[s.top]
            elif move == RIGHT and g_heads[s.i] == s.top:
                label = g_labels[s.i]
            else:
                label = 0
            paired = self.moves.pair_label_move(label, move)
            if iter_num != 0:
                self.guide.add_instance(paired, 1.0, n_feats, feats)
            self.moves.transition(move, label, &s)
            n_instances += 1
        return n_instances

    cdef int online_train_one(self, int iter_num, Sentence* sent, py_words) except -1:
        cdef size_t move, label, gold_move, gold_label, pred_move, pred_label
        cdef bint* preconditions
        cdef bint* zero_cost_moves
        cdef bint* right_arcs = self.moves.right_arcs
        cdef double weight = 1
        
        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads
        cdef size_t* tags = sent.pos

        cdef size_t* context = self._context
        cdef uint64_t* feats = self._hashed_feats
        cdef size_t n_feats = self.n_preds
        cdef State s = init_state(sent.length)
        cdef int n_instances = 0
        cdef size_t _ = 0
        cdef size_t guess_label = 0
        weight = 1
        if DEBUG:
            print ' '.join(py_words)
        n_trans = 0
        while not s.is_finished:
            n_trans += 1
            if n_trans > 500:
                raise StandardError
            features.extract(self._context, self._hashed_feats, sent, &s)
            # Determine which moves are zero-cost and meet pre-conditions
            preconditions = self.moves.check_preconditions(&s)
            pred_paired = self.guide.predict_from_ints(n_feats, feats, preconditions)
            zero_cost_moves = self.moves.check_costs(&s, g_labels, g_heads)
            if g_heads[s.top] == s.i and self.moves.allow_reattach:
                assert zero_cost_moves[self.moves.pair_label_move(g_labels[s.top], LEFT)]
            if g_heads[s.i] == s.top:
                assert zero_cost_moves[self.moves.pair_label_move(g_labels[s.i], RIGHT)]
            try:
                gold_paired = self.guide.predict_from_ints(n_feats, feats, zero_cost_moves)
            except:
                print s.stack_len
                print s.n
                print s.i
                print s.top
                print g_heads[s.top]
                print self.moves.l_cost(&s, g_heads)
                raise
            if gold_paired == pred_paired and gold_paired == self.moves.w_start:
                if DEBUG:
                    print "LOWER"
                    print py_words[s.top], s.r_valencies[s.top]
            self.guide.update(pred_paired, gold_paired, n_feats, feats, weight)
            if zero_cost_moves[pred_paired]:
                gold_paired = pred_paired
            if iter_num >= 2 and random.random() < FOLLOW_ERR_PC:
                self.moves.unpair_label_move(pred_paired, &label, &move)
            else:
                self.moves.unpair_label_move(gold_paired, &label, &move)
            # Save on reduce as well so we can use the guess label for Lower
            best_right_guess = self.guide.predict_from_ints(n_feats, feats, right_arcs)
            self.moves.unpair_label_move(best_right_guess, &guess_label, &_)
            s.guess_labels[s.top][s.i] = guess_label
            if DEBUG:
                print s.i, lmove_to_str(move, label), transition_to_str(&s, move, label, py_words)
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
        cdef uint64_t* feats = self._hashed_feats
        cdef bint* valid
        cdef bint* right_arcs = self.moves.right_arcs
        cdef size_t _ = 0
        cdef size_t guess_label = 0
        cdef size_t n = sent.length
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        while not s.is_finished:
            valid = self.moves.check_preconditions(&s)
            features.extract(context, feats, sent, &s)
            paired = self.guide.predict_from_ints(n_preds, feats, valid)
            sent.parse.moves[s.t] = paired
            sent.parse.n_moves += 1
            top = s.top
            self.moves.unpair_label_move(paired, &label, &move)
            best_right = self.guide.predict_from_ints(n_preds, feats, right_arcs)
            self.moves.unpair_label_move(best_right, &guess_label, &_)
            s.guess_labels[s.top][s.i] = guess_label
            self.moves.transition(move, label, &s)
        # No need to copy heads for root and start symbols
        for i in range(1, sent.length - 1):
            assert s.heads[i] != 0
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
            cfg.write(u'feat_thresh\t%d\n' % self.feat_thresh)
            cfg.write(u'allow_reattach\t%s\n' % self.moves.allow_reattach)
            cfg.write(u'allow_lower\t%s\n' % self.moves.allow_lower)
            cfg.write(u'allow_invert\t%s\n' % self.moves.allow_invert)
            cfg.write(u'shiftless\t%s\n' % self.moves.shiftless)
            cfg.write(u'repair_only\t%s\n' % self.moves.repair_only)
            cfg.write(u'left_labels\t%s\n' % ','.join(self.moves.left_labels))
            cfg.write(u'right_labels\t%s\n' % ','.join(self.moves.right_labels))
        
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
    cdef bint allow_invert
    cdef bint shiftless
    cdef bint repair_only
    cdef size_t n_labels
    cdef object py_labels
    cdef size_t[N_MOVES] offsets
    cdef bint* _move_validity
    cdef bint* _pair_validity
    cdef bint* right_arcs
    cdef list left_labels
    cdef list right_labels
    cdef size_t n_l_classes
    cdef size_t n_r_classes
    cdef size_t* l_classes
    cdef size_t* r_classes
    cdef size_t n_paired
    cdef size_t s_id
    cdef size_t d_id
    cdef size_t l_start
    cdef size_t l_end
    cdef size_t r_start
    cdef size_t r_end
    cdef size_t w_start
    cdef size_t w_end
    cdef size_t v_id

    cdef int n_lmoves

    def __cinit__(self, object labels, allow_reattach=False,
                  allow_lower=False, allow_invert=False,
                  shiftless=False, repair_only=False):
        self.n_labels = len(labels)
        self.py_labels = labels
        self.allow_reattach = allow_reattach
        self.allow_lower = allow_lower
        self.allow_invert = allow_invert
        self.shiftless = shiftless
        self.repair_only = repair_only
        if self.shiftless:
            assert not self.repair_only
        self.n_paired = N_MOVES * self.n_labels
        self._move_validity = <bint*>malloc(N_MOVES * sizeof(bint))
        self._pair_validity = <bint*>malloc(self.n_paired * sizeof(bint))
        self.right_arcs = <bint*>malloc(self.n_paired * sizeof(bint))
        self.s_id = SHIFT * self.n_labels
        self.d_id = REDUCE * self.n_labels
        self.l_start = LEFT * self.n_labels
        self.l_end = (LEFT + 1) * self.n_labels
        self.r_start = RIGHT * self.n_labels
        self.r_end = (RIGHT + 1) * self.n_labels
        self.w_start = LOWER * self.n_labels
        self.w_end = (LOWER + 1) * self.n_labels
        self.v_id = self.w_end + 1
        for i in range(self.n_paired):
            self.right_arcs[i] = False
        for i in range(self.r_start, self.r_end):
            self.right_arcs[i] = True

    def set_labels(self, left_labels, right_labels):
        self.left_labels = [self.py_labels[l] for l in sorted(left_labels)]
        self.right_labels = [self.py_labels[l] for l in sorted(right_labels)]
        self.l_classes = <size_t*>malloc(len(left_labels) * sizeof(size_t))
        self.r_classes = <size_t*>malloc(len(right_labels) * sizeof(size_t))
        self.n_l_classes = len(left_labels)
        self.n_r_classes = len(right_labels)
        valid_classes = [self.d_id]
        if not self.shiftless:
            valid_classes.append(self.s_id)
        for i, label in enumerate(left_labels):
            paired = self.pair_label_move(label, LEFT)
            valid_classes.append(paired)
            self.l_classes[i] = paired
        for i in range(self.n_paired):
            self.right_arcs[i] = False
        for i, label in enumerate(right_labels):
            paired = self.pair_label_move(label, RIGHT)
            valid_classes.append(paired)
            self.right_arcs[paired] = True
            self.r_classes[i] = paired
            paired = self.pair_label_move(label, LOWER)
        if self.allow_lower:
            valid_classes.append(self.w_start)
        if self.allow_invert:
            valid_classes.append(self.v_id)

        return valid_classes

    cdef size_t pair_label_move(self, size_t label, size_t move):
        return move * self.n_labels + label

    cdef int unpair_label_move(self, size_t paired, size_t* label, size_t* move):
        move[0] = paired / self.n_labels
        label[0] = paired % self.n_labels

    cdef int transition(self, size_t move, size_t label, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc
        if s.top != 0:
            assert s.second < s.top
            assert s.second != s.top
            if get_r(s, s.second) != 0:
                assert get_r(s, s.second) <= s.top, '%d of %d > %d' % (get_r(s, s.second), s.second, s.top)
        s.history[s.t] = self.pair_label_move(label, move)
        s.t += 1 
        if move == SHIFT:
            assert not self.shiftless
            push_stack(s)
        elif move == REDUCE:
            if s.heads[s.top] == 0:
                assert self.repair_only
                assert s.second != 0
                assert s.second < s.top
                add_dep(s, s.second, s.top, s.guess_labels[s.second][s.top])
            pop_stack(s)
        elif move == LEFT:
            child = pop_stack(s)
            if s.heads[child] != 0:
                assert get_r(s, s.heads[child]) == child, get_r(s, s.heads[child])
                del_r_child(s, s.heads[child])
            head = s.i
            add_dep(s, head, child, label)
            if self.shiftless and s.stack_len == 1 and not s.at_end_of_buffer:
                push_stack(s)
        elif move == RIGHT:
            child = s.i
            head = s.top
            add_dep(s, head, child, label)
            push_stack(s)
        elif move == LOWER:
            assert s.r_valencies[s.top] >= 2
            gc = get_r(s, s.top)
            c = get_r2(s, s.top)
            assert c != 0 and gc != 0
            del_r_child(s, s.top)
            assert get_r(s, s.top) == c
            add_dep(s, c, gc, s.guess_labels[c][gc])
            assert get_r(s, s.top) == c
            assert gc > c, gc
            assert get_r(s, c) == gc
            s.second = s.top
            s.top = c
            # Redundant, but just for clarity
            s.stack[s.stack_len - 1] = s.second
            s.stack[s.stack_len] = c
            s.stack_len += 1
        elif move == INVERT:
            assert s.l_valencies[s.top] >= 1
            assert self.allow_invert
            c = get_l(s, s.top)
            del_l_child(s, s.top)
            if s.heads[s.top] != 0:
                del_r_child(s, s.second)
            assert c < s.top
            add_dep(s, c, s.top, s.guess_labels[c][s.top])
            s.second = c
            s.stack[s.stack_len - 1] = c
            s.stack[s.stack_len] = s.top
            s.stack_len += 1
        else:
            raise StandardError(lmove_to_str(move, label))
        if s.i == (s.n - 1):
            s.at_end_of_buffer = True
        if s.at_end_of_buffer and s.stack_len == 1:
            s.is_finished = True

    cdef bint* check_preconditions(self, State* s) except NULL:
        cdef size_t i
        cdef bint* unpaired = self._move_validity
        # Load pre-conditions that don't refer to gold heads
        unpaired[ERR] = False
        unpaired[SHIFT] = (not s.at_end_of_buffer) and not self.shiftless
        unpaired[RIGHT] = (not s.at_end_of_buffer) and s.top != 0
        if self.repair_only:
            unpaired[REDUCE] = s.heads[s.top] != 0 or s.second != 0
        else:
            unpaired[REDUCE] = s.heads[s.top] != 0
        if self.shiftless and unpaired[REDUCE]:
            assert s.stack_len >= 2
        unpaired[LEFT] = s.top != 0 and (s.heads[s.top] == 0 or self.allow_reattach)
        unpaired[LOWER] = self.allow_lower and s.r_valencies[s.top] >= 2
        unpaired[INVERT] = self.allow_invert and s.l_valencies[s.top] >= 1
        cdef bint* paired = self._pair_validity
        for i in range(self.n_paired):
            paired[i] = False
        paired[self.s_id] = unpaired[SHIFT]
        paired[self.d_id] = unpaired[REDUCE]
        for i in range(self.n_l_classes):
            paired[self.l_classes[i]] = unpaired[LEFT]
        for i in range(self.n_r_classes):
            paired[self.r_classes[i]] = unpaired[RIGHT]
        paired[self.w_start] = unpaired[LOWER]
        paired[self.v_id] = unpaired[INVERT]
        return paired
   
    cdef bint* check_costs(self, State* s, size_t* labels, size_t* heads) except NULL:
        cdef size_t l_id, r_id, w_id
        cdef bint* valid_moves = self._move_validity
        paired_validity = self.check_preconditions(s)
        valid_moves[SHIFT] = valid_moves[SHIFT] and self.s_cost(s, heads)
        valid_moves[REDUCE] = valid_moves[REDUCE] and self.d_cost(s, heads)
        valid_moves[LEFT] = valid_moves[LEFT] and self.l_cost(s, heads)
        valid_moves[RIGHT] = valid_moves[RIGHT] and self.r_cost(s, heads)
        valid_moves[LOWER] = valid_moves[LOWER] and self.w_cost(s, heads)
        valid_moves[INVERT] = valid_moves[INVERT] and self.v_cost(s, heads)
        for i in range(self.n_paired):
            paired_validity[i] = False
        paired_validity[self.s_id] = valid_moves[SHIFT]
        paired_validity[self.d_id] = valid_moves[REDUCE]
        paired_validity[self.w_start] = valid_moves[LOWER]
        paired_validity[self.v_id] = valid_moves[INVERT]
        if valid_moves[LEFT] and heads[s.top] == s.i:
            paired_validity[self.pair_label_move(labels[s.top], LEFT)] = True
        else:
            for i in range(self.n_r_classes):
                paired_validity[self.l_classes[i]] = valid_moves[LEFT]
        if valid_moves[RIGHT] and heads[s.i] == s.top:
            paired_validity[self.pair_label_move(labels[s.i], RIGHT)] = True
        else:
            for i in range(self.n_r_classes):
                paired_validity[self.r_classes[i]] = valid_moves[RIGHT]
        return paired_validity

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
            assert not self.shiftless
            return SHIFT
        if self._move_validity[LOWER]:
            return LOWER
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
            # If right-arcing the word provides a path to the correct head via Lower,
            # don't shift.
            if has_head_via_lower(s, s.i, g_heads):
                return False
            #for i in range(1, s.stack_len):
            #    stack_i = s.stack[i]
            #    if get_r(s, stack_i) != 0 and g_heads[s.i] == get_r(s, stack_i):
            #        return False

            # Why's this only apply to top again? I think there was a good
            # reason, I remember fixing this.
            if s.r_valencies[s.top] >= 2 and g_heads[get_r(s, s.top)] == get_r2(s, s.top) \
              and g_heads[s.i] == get_r2(s, s.top):
                return False
        return True

    cdef bint r_cost(self, State *s, size_t* g_heads):
        cdef size_t i, buff_i, stack_i
        if g_heads[s.i] == s.top:
            return True
        if has_head_in_buffer(s, s.i, g_heads):
            if self.repair_only:
                return False
            elif not self.allow_reattach:
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
            if s.r_valencies[s.top] >= 2 and g_heads[get_r(s, s.top)] == get_r2(s, s.top):
                return False
        return True

    cdef bint d_cost(self, State *s, size_t* g_heads):
        if has_child_in_buffer(s, s.top, g_heads):
            if self.repair_only:
                return False
            if not self.allow_lower:
                return False
            elif s.second == 0 or s.heads[s.top] != s.second:
                return False
        if self.allow_reattach and has_head_in_buffer(s, s.top, g_heads):
            return False
        if self.allow_reattach:
            assert g_heads[s.top] != s.n and g_heads[s.top] != (s.n - 1)
        if self.allow_lower and get_r(s, s.top) != 0:
            if has_grandchild_via_lower(s, s.i, g_heads):
                return False
            if s.r_valencies[s.top] >= 2 and g_heads[get_r(s, s.top)] == get_r2(s, s.top):
                return False
        if self.allow_invert and s.top != 0 and g_heads[s.top] == get_l(s, s.top):
            return False
        return True

    cdef bint l_cost(self, State *s, size_t* g_heads):
        cdef size_t buff_i
        if g_heads[s.top] == s.i:
            return True
        if has_head_in_buffer(s, s.top, g_heads):
            return False
        assert g_heads[s.top] != s.n and g_heads[s.top] != (s.n - 1)
        if has_child_in_buffer(s, s.top, g_heads):
            return False
        if self.allow_reattach and g_heads[s.top] == s.heads[s.top]:
            return False
        if self.repair_only and g_heads[s.top] == s.second:
            return False
        if self.allow_lower:
            for buff_i in range(s.i, s.n - 1):
                if g_heads[buff_i] == get_r(s, s.top):
                    return False
            if s.r_valencies[s.top] >= 2 and g_heads[get_r(s, s.top)] == get_r2(s, s.top):
                return False
        if self.allow_invert and s.top != 0 and g_heads[s.top] == get_l(s, s.top):
            return False
        return True

    cdef bint w_cost(self, State *s, size_t* g_heads):
        r = get_r(s, s.top)
        r2 = get_r2(s, s.top)
        assert r != 0 and r2 != 0
        #if g_heads[r] == r2:
        #    return True
        #return False
        # TODO: Is this a good idea? It doesnt seem to work
        # Other functions were depending on this. Might work now.
        if g_heads[r] == s.heads[r]:
            return False
        return True

    cdef bint v_cost(self, State *s, size_t* g_heads):
        left = get_l(s, s.top)
        assert left != 0
        #if g_heads[s.top] == left:
        #    return True
        #return False
        # TODO: Is this a good idea? Is it causing variance?
        if g_heads[left] == s.top:
            return False
        if g_heads[s.top] == s.heads[s.top]:
            return False
        # ie if reduce-reattach is allowed
        if self.repair_only and g_heads[s.top] == s.second:
            return False
        if has_head_in_buffer(s, s.top, g_heads):
            if not (self.allow_reattach and not self.repair_only):
                return False
        return True

cdef transition_to_str(State* s, size_t move, label, object tokens):
    tokens = tokens + ['<end>']
    if move == SHIFT:
        return u'%s-->%s' % (tokens[s.i], tokens[s.top])
    elif move == REDUCE:
        if s.heads[s.top] == 0:
            return u'%s(%s)!!' % (tokens[s.second], tokens[s.top])
        return u'%s/%s' % (tokens[s.top], tokens[s.second])
    elif move == LOWER:
        child = tokens[get_r(s, s.top)]
        parent = tokens[get_r2(s, s.top)]
        top = tokens[s.top]
        return u'%s(%s, %s) ---> %s(%s(%s))' % (top, parent, child, top, parent, child)
    elif move == INVERT:
        left = tokens[get_l(s, s.top)]
        top = tokens[s.top]
        return u'%s(%s) --> %s(%s)' % (top, left, left, top)
    else:
        if move == LEFT:
            head = s.i
            child = s.top
        else:
            head = s.top
            child = s.i if s.i < len(tokens) else 0
        return u'%s(%s)' % (tokens[head], tokens[child])



