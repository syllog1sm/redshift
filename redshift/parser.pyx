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
    _n_moves

DEF N_MOVES = 5
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
                  add_extra=True, label_set='MALT', feat_thresh=5,
                  allow_reattach=False, allow_reduce=False,
                  reuse_idx=False):
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
            allow_reduce = params['allow_reduce'] == 'True'
            l_labels = params['left_labels']
            r_labels = params['right_labels']
        if allow_reattach and allow_reduce:
            print 'NM L+D'
        elif allow_reattach:
            print 'NM L'
        elif allow_reduce:
            print 'NM D'
        else:
            print 'Baseline'
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
        self.moves = TransitionSystem(io_parse.LABEL_STRS, allow_reattach=allow_reattach,
                                      allow_reduce=allow_reduce)
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
        for i in range(sents.length):
            sent = &sents.s[i]
            for j in range(1, sent.length - 1):
                label = sent.parse.labels[j]
                if sent.parse.heads[j] > j:
                    seen_l_labels.add(label)
                else:
                    seen_r_labels.add(label)
        move_classes = self.moves.set_labels(seen_l_labels, seen_r_labels)
        self.guide.set_classes(move_classes)
        self.write_cfg(self.model_dir.join('parser.cfg'))
        indices = range(sents.length)
        for n in range(n_iter):
            random.shuffle(indices)
            for sent_id, i in enumerate(indices):
                if self.train_alg == 'online':
                    self.online_train_one(n, &sents.s[i], sents.strings[i][0])
                else:
                    self.static_train_one(n, &sents.s[i], sents.strings[i][0])
            move_acc = (float(self.guide.n_corr) / self.guide.total) * 100
            print "#%d: Moves %d/%d=%.2f" % (n, self.guide.n_corr,
                                             self.guide.total, move_acc)
            if self.feat_thresh > 1:
                self.guide.prune(self.feat_thresh)
            self.guide.n_corr = 0
            self.guide.total = 0
        self.guide.train()

    cdef int static_train_one(self, size_t iter_num, Sentence* sent, object py_words) except -1:
        cdef size_t clas, move, label

        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads
        cdef bint* right_arcs = self.moves.right_arcs
        cdef bint* left_arcs = self.moves.left_arcs
        cdef uint64_t* feats = self._hashed_feats
        cdef size_t* context = self._context
        cdef int n_feats = self.n_preds

        cdef State s = init_state(sent.length)
        cdef int n_instances = 0
        if DEBUG:
            print ' '.join(py_words)
        while not s.is_finished:
            features.extract(context, feats, sent, &s)
            move = self.moves.move_from_gold(&s, g_labels, g_heads)
            if move == LEFT:
                assert g_heads[s.top] == s.i
                label = g_labels[s.top]
            elif move == RIGHT:
                assert g_heads[s.i] == s.top
                label = g_labels[s.i]
            else:
                label = 0
            clas = self.moves.pair_label_move(label, move)
            self.guide.add_instance(clas, 1.0, n_feats, feats)
            if DEBUG:
                print s.i, lmove_to_str(move, label), transition_to_str(&s, move, label, py_words)
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
        while not s.is_finished:
            features.extract(self._context, self._hashed_feats, sent, &s)
            # Determine which moves are zero-cost and meet pre-conditions
            preconditions = self.moves.check_preconditions(&s)
            pred_paired = self.guide.predict_from_ints(n_feats, feats, preconditions)
            zero_cost_moves = self.moves.check_costs(&s, g_labels, g_heads)
            gold_paired = self.guide.predict_from_ints(n_feats, feats, zero_cost_moves)
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
        cdef size_t move = 0
        cdef size_t label = 0
        cdef size_t n_preds = self.n_preds
        cdef size_t* context = self._context
        cdef uint64_t* feats = self._hashed_feats
        cdef bint* valid
        cdef double* scores
        cdef size_t _ = 0
        cdef size_t guess_label = 0
        cdef size_t n = sent.length
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        while not s.is_finished:
            features.extract(context, feats, sent, &s)
            valid = self.moves.check_preconditions(&s)
            self.predict(n_preds, feats, valid, &move, &label,
                         &s.guess_labels[s.top][s.i])
            sent.parse.moves[s.t] = self.moves.pair_label_move(label, move)
            self.moves.transition(move, label, &s)
            sent.parse.n_moves += 1
        # No need to copy heads for root and start symbols
        for i in range(1, sent.length - 1):
            assert s.heads[i] != 0
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]

    cdef int predict(self, uint64_t n_preds, uint64_t* feats, bint* valid,
            size_t* move, size_t* label, size_t* rlabel) except -1:
        cdef:
            size_t i
            double score
            size_t clas, best_valid, best_right
            double* scores

        cdef size_t right_move = 0
        cdef double valid_score = -1000000
        cdef double right_score = -100000
        cdef uint64_t* labels = self.guide.get_labels()
        scores = self.guide.predict_scores(n_preds, feats)
        seen_valid = False
        for i in range(self.guide.nr_class):
            score = scores[i]
            clas = labels[i]
            if valid[clas] and score > valid_score:
                best_valid = clas
                valid_score = score
                seen_valid = True
            if self.moves.right_arcs[clas] and score > right_score:
                best_right = clas
                right_score = score
        assert seen_valid
        self.moves.unpair_label_move(best_valid, label, move)
        self.moves.unpair_label_move(best_right, rlabel, &right_move)

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
            cfg.write(u'allow_reduce\t%s\n' % self.moves.allow_reduce)
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
    cdef bint allow_reduce
    cdef size_t n_labels
    cdef object py_labels
    cdef size_t[N_MOVES] offsets
    cdef bint* _move_validity
    cdef bint* _pair_validity
    cdef bint* right_arcs
    cdef bint* left_arcs
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
                  allow_reduce=False):
        self.n_labels = len(labels)
        self.py_labels = labels
        self.allow_reattach = allow_reattach
        self.allow_reduce = allow_reduce
        self.n_paired = N_MOVES * self.n_labels
        self._move_validity = <bint*>malloc(N_MOVES * sizeof(bint))
        self._pair_validity = <bint*>malloc(self.n_paired * sizeof(bint))
        self.right_arcs = <bint*>malloc(self.n_paired * sizeof(bint))
        self.left_arcs = <bint*>malloc(self.n_paired * sizeof(bint))
        self.s_id = SHIFT * self.n_labels
        self.d_id = REDUCE * self.n_labels
        self.l_start = LEFT * self.n_labels
        self.l_end = (LEFT + 1) * self.n_labels
        self.r_start = RIGHT * self.n_labels
        self.r_end = (RIGHT + 1) * self.n_labels
        for i in range(self.n_paired):
            self.right_arcs[i] = False
            self.left_arcs[i] = False
        for i in range(self.r_start, self.r_end):
            self.right_arcs[i] = True
        for i in range(self.l_start, self.l_end):
            self.left_arcs[i] = True

    def set_labels(self, left_labels, right_labels):
        self.left_labels = [self.py_labels[l] for l in sorted(left_labels)]
        self.right_labels = [self.py_labels[l] for l in sorted(right_labels)]
        self.l_classes = <size_t*>malloc(len(left_labels) * sizeof(size_t))
        self.r_classes = <size_t*>malloc(len(right_labels) * sizeof(size_t))
        self.n_l_classes = len(left_labels)
        self.n_r_classes = len(right_labels)
        valid_classes = [self.d_id]
        valid_classes.append(self.s_id)
        for i in range(self.n_paired):
            self.right_arcs[i] = False
            self.left_arcs[i] = False
        for i, label in enumerate(left_labels):
            paired = self.pair_label_move(label, LEFT)
            valid_classes.append(paired)
            self.l_classes[i] = paired
            self.left_arcs[paired] = True
        for i, label in enumerate(right_labels):
            paired = self.pair_label_move(label, RIGHT)
            valid_classes.append(paired)
            self.right_arcs[paired] = True
            self.r_classes[i] = paired
        return valid_classes

    cdef size_t pair_label_move(self, size_t label, size_t move):
        return move * self.n_labels + label

    cdef int unpair_label_move(self, size_t paired, size_t* label, size_t* move):
        move[0] = paired / self.n_labels
        label[0] = paired % self.n_labels

    cdef int transition(self, size_t move, size_t label, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc
        s.history[s.t] = self.pair_label_move(label, move)
        s.t += 1 
        if move == SHIFT:
            push_stack(s)
        elif move == REDUCE:
            if s.heads[s.top] == 0:
                assert self.allow_reduce
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
        elif move == RIGHT:
            child = s.i
            head = s.top
            add_dep(s, head, child, label)
            push_stack(s)
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
        unpaired[SHIFT] = not s.at_end_of_buffer
        unpaired[RIGHT] = (not s.at_end_of_buffer) and s.top != 0
        if self.allow_reduce:
            unpaired[REDUCE] = s.heads[s.top] != 0 or s.second != 0
        else:
            unpaired[REDUCE] = s.heads[s.top] != 0
        unpaired[LEFT] = s.top != 0 and (s.heads[s.top] == 0 or self.allow_reattach)
        cdef bint* paired = self._pair_validity
        for i in range(self.n_paired):
            paired[i] = False
        paired[self.s_id] = unpaired[SHIFT]
        paired[self.d_id] = unpaired[REDUCE]
        for i in range(self.n_l_classes):
            paired[self.l_classes[i]] = unpaired[LEFT]
        for i in range(self.n_r_classes):
            paired[self.r_classes[i]] = unpaired[RIGHT]
        return paired
   
    cdef bint* check_costs(self, State* s, size_t* labels, size_t* heads) except NULL:
        cdef size_t l_id, r_id, w_id
        cdef bint* valid_moves = self._move_validity
        paired_validity = self.check_preconditions(s)
        valid_moves[SHIFT] = valid_moves[SHIFT] and self.s_cost(s, heads)
        valid_moves[REDUCE] = valid_moves[REDUCE] and self.d_cost(s, heads)
        valid_moves[LEFT] = valid_moves[LEFT] and self.l_cost(s, heads)
        valid_moves[RIGHT] = valid_moves[RIGHT] and self.r_cost(s, heads)
        for i in range(self.n_paired):
            paired_validity[i] = False
        paired_validity[self.s_id] = valid_moves[SHIFT]
        paired_validity[self.d_id] = valid_moves[REDUCE]
        if valid_moves[LEFT] and heads[s.top] == s.i:
            paired_validity[self.pair_label_move(labels[s.top], LEFT)] = True
        else:
            for i in range(self.n_l_classes):
                paired_validity[self.l_classes[i]] = valid_moves[LEFT]
        if valid_moves[RIGHT] and heads[s.i] == s.top:
            paired_validity[self.pair_label_move(labels[s.i], RIGHT)] = True
        else:
            for i in range(self.n_r_classes):
                paired_validity[self.r_classes[i]] = valid_moves[RIGHT]
        return paired_validity

    cdef int move_from_gold(self, State* s, size_t* labels, size_t* heads) except -1:
        if s.stack_len == 1:
            return SHIFT
        if not s.at_end_of_buffer and heads[s.i] == s.top:
            return RIGHT
        elif heads[s.top] == s.i and (self.allow_reattach or s.heads[s.top] == 0):
            return LEFT
        elif self.d_cost(s, heads) and (self.allow_reduce or s.heads[s.top] != 0):
            return REDUCE
        elif self.s_cost(s, heads):
            return SHIFT
        else:
            raise StandardError

    cdef bint s_cost(self, State *s, size_t* g_heads):
        cdef size_t i, stack_i
        if has_child_in_stack(s, s.i, g_heads):
            return False
        if has_head_in_stack(s, s.i, g_heads):
            return False
        return True

    cdef bint r_cost(self, State *s, size_t* g_heads):
        cdef size_t i, buff_i, stack_i
        if g_heads[s.i] == s.top:
            return True
        if has_head_in_buffer(s, s.i, g_heads):
            return False
        if has_child_in_stack(s, s.i, g_heads):
            return False
        if has_head_in_stack(s, s.i, g_heads):
            return False
        return True

    cdef bint d_cost(self, State *s, size_t* g_heads):
        if has_child_in_buffer(s, s.top, g_heads):
            return False
        if has_head_in_buffer(s, s.top, g_heads):
            if self.allow_reattach:
                return False
            if self.allow_reduce and s.heads[s.top] == 0:
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
        if self.allow_reduce and g_heads[s.top] == s.second:
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
    else:
        if move == LEFT:
            head = s.i
            child = s.top
        else:
            head = s.top
            child = s.i if s.i < len(tokens) else 0
        return u'%s(%s)' % (tokens[head], tokens[child])



