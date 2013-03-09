# cython: profile=True
"""
MALT-style dependency parser
"""
cimport cython
import random
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy
from pathlib import Path
from collections import defaultdict
import sh
import sys
from itertools import izip

from _state cimport *
cimport io_parse
import io_parse
from io_parse cimport Sentence
from io_parse cimport Sentences
from io_parse cimport make_sentence
cimport features
from features cimport FeatureSet

from io_parse import LABEL_STRS, STR_TO_LABEL

import index.hashes
cimport index.hashes

from svm.cy_svm cimport Model, LibLinear, Perceptron

from libc.stdint cimport uint64_t, int64_t
from libc.stdlib cimport qsort

from _state cimport *

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
    cdef FeatureSet features
    cdef Perceptron guide
    cdef object model_dir
    cdef size_t beam_width
    cdef TransitionSystem moves
    cdef object add_extra
    cdef object label_set
    cdef object train_alg
    cdef int feat_thresh

    cdef bint label_beam
    cdef object upd_strat

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  add_extra=True, label_set='MALT', feat_thresh=5,
                  allow_reattach=False, allow_reduce=False,
                  reuse_idx=False, beam_width=1, upd_strat="early", 
                  label_beam=True):
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
            beam_width = int(params['beam_width'])
            upd_strat = params['upd_strat']
            label_beam = params['label_beam'] == 'True'
        if allow_reattach and allow_reduce:
            print 'NM L+D'
        elif allow_reattach:
            print 'NM L'
        elif allow_reduce:
            print 'NM D'
        if beam_width >= 1:
            beam_settings = (beam_width, upd_strat, label_beam)
            print 'Beam settings: k=%d; upd_strat=%s; label_beam=%s' % beam_settings
        self.model_dir = self.setup_model_dir(model_dir, clean)
        labels = io_parse.set_labels(label_set)
        self.features = FeatureSet(len(labels), add_extra)
        self.add_extra = add_extra
        self.label_set = label_set
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        self.beam_width = beam_width
        self.upd_strat = upd_strat
        self.label_beam = label_beam
        if clean == True:
            self.new_idx(self.model_dir, self.features.n)
        else:
            self.load_idx(self.model_dir, self.features.n)
        self.moves = TransitionSystem(labels, allow_reattach=allow_reattach,
                                      allow_reduce=allow_reduce)
        if not clean:
            self.moves.set_labels(_parse_labels_str(l_labels), _parse_labels_str(r_labels))
        guide_loc = self.model_dir.join('model')
        n_labels = len(io_parse.LABEL_STRS)
        self.guide = Perceptron(self.moves.max_class, guide_loc)

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
        self.guide.set_classes(range(move_classes))
        self.write_cfg(self.model_dir.join('parser.cfg'))
        indices = range(sents.length)
        if self.beam_width >= 1:
            self.guide.use_cache = True
        self.features.save_entries = True
        stats = defaultdict(int)
        for n in range(n_iter):
            random.shuffle(indices)
            # Group indices into minibatches of fixed size
            for minibatch in izip(*[iter(indices)] * 1):
                deltas = []
                for i in minibatch:
                    if self.beam_width >= 1:
                        deltas.append(self.decode_beam(&sents.s[i], self.beam_width,
                                      stats))
                    else:
                        self.train_one(n, &sents.s[i], sents.strings[i][0])
                for weights in deltas:
                    self.guide.batch_update(weights)
            print_train_msg(n, self.guide.n_corr, self.guide.total,
                            self.guide.cache.n_hit, self.guide.cache.n_miss,
                            stats)
            if self.feat_thresh > 1:
                self.guide.prune(self.feat_thresh)
            self.guide.n_corr = 0
            self.guide.total = 0
        self.guide.train()

    cdef dict decode_beam(self, Sentence* sent, size_t k, object stats):
        cdef size_t i
        cdef int* costs
        cdef int cost
        cdef size_t* g_heads = sent.parse.heads
        cdef size_t* g_labels = sent.parse.labels
        cdef State * s
        cdef State* parent
        cdef Cont* cont
        cdef Violation violn
        cdef bint halt = False
        self.guide.cache.flush()
        cdef Beam beam = Beam(k, sent.length, self.guide.nr_class,
                              upd_strat=self.upd_strat, add_labels=self.label_beam)
        stats['sents'] += 1
        while not beam.gold.is_finished:
            beam.refresh()
            self._fill_move_scores(sent, beam.psize, beam.parents, beam.next_moves)
            beam.sort_moves()
            self._advance_gold(beam.gold, sent, self.train_alg == 'static')
            for i in range(beam.nr_class):
                cont = &beam.next_moves[i]
                if not cont.is_valid:
                    continue
                if not beam.accept(cont.parent, self.moves.moves[cont.clas], cont.score):
                    continue
                parent = beam.parents[cont.parent]
                if self.train_alg == 'static':
                    cost = cont.clas != self.moves.break_tie(parent, g_heads, g_labels)
                else:
                    costs = self.moves.get_costs(parent, g_heads, g_labels)
                    cost = costs[cont.clas]
                    assert cost != -1
                s = beam.add(cont.parent, cont.score, cost)
                self.moves.transition(cont.clas, s)
                if beam.is_full:
                    break
            assert beam.bsize
            if beam.bsize:
                assert beam.gold.t == beam.beam[0].t, '%d vs %d (%d)'
            assert beam.bsize <= beam.k, beam.bsize
            halt = beam.check_violation()
            if halt:
                stats['early'] += 1
                break
            elif beam.beam[0].is_gold:
                self.guide.n_corr += 1
            for i in range(beam.bsize):
                assert beam.beam[i].t == beam.gold.t, '%d vs %d' % (beam.beam[i].t, beam.gold.t)
        self.guide.total += beam.gold.t
        if beam.first_violn is not None:
            violn = beam.pick_violation()
            stats['moves'] += violn.t
            counted = self._count_feats(sent, violn.t, violn.phist, violn.ghist)
            return counted
        else:
            stats['moves'] += beam.gold.t
            return {}

    cdef int _advance_gold(self, State* s, Sentence* sent, bint use_static) except -1:
        cdef:
            size_t oracle, i
            int* costs
            uint64_t* feats
            double* scores
            bint cache_hit
            double best_score
        fill_kernel(s)
        scores = self.guide.cache.lookup(sizeof(s.kernel), <void*>&s.kernel, &cache_hit)
        if not cache_hit:
            feats = self.features.extract(sent, s)
            self.guide.model.get_scores(self.features.n, feats, scores)
        if use_static:
            oracle = self.moves.break_tie(s, sent.parse.heads, sent.parse.labels)
        else:
            costs = self.moves.get_costs(s, sent.parse.heads, sent.parse.labels)
            best_score = -10000
            for i in range(self.moves.nr_class):
                if costs[i] == 0 and scores[i] > best_score:
                    oracle = i
                    best_score = scores[i]
            assert best_score > -10000
        s.score += scores[oracle]
        self.moves.transition(oracle, s)

    cdef int _fill_move_scores(self, Sentence* sent, size_t k, State** parents,
            Cont* next_moves) except -1:
        cdef size_t parent_idx, child_idx
        cdef State* parent
        cdef uint64_t* feats
        cdef int* valid
        cdef size_t n_feats = self.features.n
        cdef bint cache_hit = False
        cdef size_t move_idx = 0
        cdef size_t n_valid = 0
        for parent_idx in range(k):
            parent = parents[parent_idx]
            fill_kernel(parent)
            scores = self.guide.cache.lookup(sizeof(parent.kernel),
                    <void*>&parent.kernel, &cache_hit)
            if not cache_hit:
                feats = self.features.extract(sent, parent)
                self.guide.model.get_scores(n_feats, feats, scores)
            valid = self.moves.get_valid(parent)
            for child_idx in range(self.guide.nr_class):
                next_moves[move_idx].parent = parent_idx
                next_moves[move_idx].clas = child_idx
                if valid[child_idx] == 1:
                    parent.nr_kids += 1
                    next_moves[move_idx].score = scores[child_idx] + parent.score
                    next_moves[move_idx].is_valid = True
                    n_valid += 1
                else:
                    next_moves[move_idx].score = -10001
                    next_moves[move_idx].is_valid = False
                move_idx += 1
        assert n_valid != 0
        return n_valid

    cdef dict _count_feats(self, Sentence* sent, size_t t, size_t* phist, size_t* ghist):
        cdef size_t d, i, f
        cdef size_t n_feats = self.features.n
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(sent.length, self.moves.n_labels)
        cdef State* pred_state = init_state(sent.length, self.moves.n_labels)
        # Find where the states diverge
        for d in range(t):
            if ghist[d] == phist[d]:
                self.moves.transition(ghist[d], gold_state)
                self.moves.transition(phist[d], pred_state)
            else:
                break
        cdef dict counts = {}
        for i in range(d, t):
            feats = self.features.extract(sent, gold_state)
            clas = ghist[i]
            counts.setdefault(clas, {})
            for f in range(n_feats):
                if feats[f] == 0:
                    break
                counts[clas].setdefault(feats[f], 0)
                counts[clas][feats[f]] += 1
            self.moves.transition(clas, gold_state)
        free_state(gold_state)
        for i in range(d, t):
            feats = self.features.extract(sent, pred_state)
            clas = phist[i]
            counts.setdefault(clas, {})
            for f in range(n_feats):
                if feats[f] == 0:
                    break
                counts[clas].setdefault(feats[f], 0)
                counts[clas][feats[f]] -= 1
            self.moves.transition(clas, pred_state)
        free_state(pred_state)
        return counts

    cdef int train_one(self, int iter_num, Sentence* sent, py_words) except -1:
        cdef int* valid
        cdef int* costs
        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads

        cdef size_t n_feats = self.features.n
        cdef State* s = init_state(sent.length, self.moves.n_labels)
        cdef size_t move = 0
        cdef size_t label = 0
        cdef size_t _ = 0
        cdef bint online = self.train_alg == 'online'
        if DEBUG:
            print ' '.join(py_words)
        while not s.is_finished:
            feats = self.features.extract(sent, s)
            valid = self.moves.get_valid(s)
            pred = self.predict(n_feats, feats, valid, &s.guess_labels[s.i])
            if online:
                costs = self.moves.get_costs(s, g_heads, g_labels)
                gold = self.predict(n_feats, feats, costs, &_) if costs[pred] != 0 else pred
            else:
                gold = self.moves.break_tie(s, g_heads, g_labels)
            self.guide.update(pred, gold, n_feats, feats, 1)
            if online and iter_num >= 2 and random.random() < FOLLOW_ERR_PC:
                self.moves.transition(pred, s)
            else:
                self.moves.transition(gold, s)
        free_state(s)

    def add_parses(self, Sentences sents, Sentences gold=None, k=None):
        cdef:
            size_t i
        if k == None:
            k = self.beam_width
        for i in range(sents.length):
            if k <= 1:
                self.parse(&sents.s[i])
            else:
                self.beam_parse(&sents.s[i], k)
        if gold is not None:
            return sents.evaluate(gold)

    cdef int parse(self, Sentence* sent) except -1:
        cdef State* s
        cdef size_t move = 0
        cdef size_t label = 0
        cdef size_t clas
        cdef size_t n_preds = self.features.n
        cdef uint64_t* feats
        cdef double* scores
        s = init_state(sent.length, self.moves.n_labels)
        sent.parse.n_moves = 0
        while not s.is_finished:
            feats = self.features.extract(sent, s)
            clas = self.predict(n_preds, feats, self.moves.get_valid(s),
                                  &s.guess_labels[s.i])
            sent.parse.moves[s.t] = clas
            self.moves.transition(clas, s)
        sent.parse.n_moves = s.t
        # No need to copy heads for root and start symbols
        for i in range(1, sent.length - 1):
            assert s.heads[i] != 0
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]
        free_state(s)
    
    cdef int beam_parse(self, Sentence* sent, size_t k) except -1:
        cdef size_t i, c, n_valid
        cdef State* s
        cdef State* new
        cdef Cont* cont
        cdef Beam beam = Beam(k, sent.length, self.guide.nr_class)
        self.guide.cache.flush()
        while not beam.beam[0].is_finished:
            beam.refresh()
            n_valid = self._fill_move_scores(sent, beam.psize, beam.parents,
                                             beam.next_moves)
            beam.sort_moves()
            for c in range(n_valid):
                cont = &beam.next_moves[c]
                if not beam.accept(cont.parent, self.moves.moves[cont.clas], cont.score):
                    continue
                s = beam.add(cont.parent, cont.score, False)
                self.moves.transition(cont.clas, s)
                if beam.is_full:
                    break
        s = beam.best_p()
        sent.parse.n_moves = s.t
        for i in range(s.t):
            sent.parse.moves[i] = s.history[i]
        # No need to copy heads for root and start symbols
        for i in range(1, sent.length - 1):
            assert s.heads[i] != 0
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]

    cdef int predict(self, uint64_t n_preds, uint64_t* feats, bint* valid,
                     size_t* rlabel) except -1:
        cdef:
            size_t i
            double score
            size_t clas, best_valid, best_right
            double* scores

        cdef size_t right_move = 0
        cdef double valid_score = -1000000
        cdef double right_score = -1000000
        scores = self.guide.predict_scores(n_preds, feats)
        seen_valid = False
        for clas in range(self.guide.nr_class):
            score = scores[clas]
            if valid[clas] and score > valid_score:
                best_valid = clas
                valid_score = score
                seen_valid = True
            if self.moves.r_end > clas >= self.moves.r_start and score > right_score:
                best_right = clas
                right_score = score
        assert seen_valid
        rlabel[0] = self.moves.labels[best_right]
        return best_valid

    def save(self):
        self.guide.save(self.model_dir.join('model'))
        self.features.save(self.model_dir.join('features'))

    def load(self):
        self.guide.load(self.model_dir.join('model'))

    def new_idx(self, model_dir, size_t n_predicates):
        #index.hashes.init_feat_idx(n_predicates, model_dir.join('features'))
        index.hashes.init_word_idx(model_dir.join('words'))
        index.hashes.init_pos_idx(model_dir.join('pos'))

    def load_idx(self, model_dir, size_t n_predicates):
        model_dir = Path(model_dir)
        index.hashes.load_word_idx(model_dir.join('words'))
        index.hashes.load_pos_idx(model_dir.join('pos'))
        self.features.load(model_dir.join('features'))
        #index.hashes.load_feat_idx(n_predicates, model_dir.join('features'))
   
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
            cfg.write(u'beam_width\t%d\n' % self.beam_width)
            cfg.write(u'upd_strat\t%s\n' % self.upd_strat)
            cfg.write(u'label_beam\t%s\n' % self.label_beam)
        
    def get_best_moves(self, Sentences sents, Sentences gold):
        """Get a list of move taken/oracle move pairs for output"""
        cdef State* s
        cdef size_t n
        cdef size_t move = 0
        cdef size_t label = 0
        cdef object best_moves
        cdef size_t i
        cdef int* costs
        cdef size_t* g_labels
        cdef size_t* g_heads
        cdef size_t clas, parse_class
        best_moves = []
        for i in range(sents.length):
            sent = &sents.s[i]
            g_labels = gold.s[i].parse.labels
            g_heads = gold.s[i].parse.heads
            n = sent.length
            s = init_state(n, self.moves.n_labels)
            sent_moves = []
            tokens = sents.strings[i][0]
            while not s.is_finished:
                costs = self.moves.get_costs(s, g_heads, g_labels)
                best_strs = []
                best_ids = set()
                for clas in range(self.moves.nr_class):
                    if costs[clas] == 0:
                        move = self.moves.moves[clas]
                        label = self.moves.labels[clas]
                        if move not in best_ids:
                            best_strs.append(lmove_to_str(move, label))
                        best_ids.add(move)
                best_strs = ','.join(best_strs)
                best_id_str = ','.join(map(str, sorted(best_ids)))
                parse_class = sent.parse.moves[s.t]
                state_str = transition_to_str(s, self.moves.moves[parse_class],
                                              self.moves.labels[parse_class],
                                              tokens)
                parse_move_str = lmove_to_str(move, label)
                if move not in best_ids:
                    parse_move_str = red(parse_move_str)
                sent_moves.append((best_id_str, int(move),
                                  best_strs, parse_move_str,
                                  state_str))
                self.moves.transition(parse_class, s)
            free_state(s)
            best_moves.append((u' '.join(tokens), sent_moves))
        return best_moves


cdef class Beam:
    cdef State** parents
    cdef State** beam
    cdef Cont* next_moves
    cdef State* gold
    cdef size_t n_labels
    cdef size_t nr_class
    cdef size_t k
    cdef size_t bsize
    cdef size_t psize
    cdef Violation first_violn
    cdef Violation max_violn
    cdef Violation last_violn
    cdef Violation cost_violn
    cdef bint is_full
    cdef bint early_upd
    cdef bint max_upd
    cdef bint late_upd
    cdef bint cost_upd
    cdef bint add_labels
    cdef bint** seen_moves

    def __cinit__(self, size_t k, size_t length, size_t nr_class, upd_strat='early',
                  add_labels=True):
        cdef size_t i
        cdef Cont* cont
        cdef State* s
        self.n_labels = len(io_parse.LABEL_STRS)
        self.k = k
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        for i in range(k):
            self.parents[i] = init_state(length, self.n_labels)
        for i in range(k):
            self.beam[i] = init_state(length, self.n_labels)
        self.gold = init_state(length, self.n_labels)
        self.bsize = 1
        self.psize = 0
        self.is_full = self.bsize >= self.k
        self.nr_class = nr_class * k
        self.next_moves = <Cont*>malloc(self.nr_class * sizeof(Cont))
        self.seen_moves = <bint**>malloc(self.nr_class * sizeof(bint*))
        for i in range(self.nr_class):
            self.next_moves[i] = Cont(score=-10000, clas=0, parent=0, is_gold=True, is_valid=True)
            self.seen_moves[i] = <bint*>calloc(N_MOVES, sizeof(bint))
        self.first_violn = None
        self.max_violn = None
        self.last_violn = None
        self.early_upd = False
        self.cost_upd = False
        self.add_labels = add_labels
        self.late_upd = False
        self.max_upd = False
        self.cost_upd = False
        if upd_strat == 'early':
            self.early_upd = True
        elif upd_strat == 'late':
            self.late_upd = True
        elif upd_strat == 'max':
            self.max_upd = True
        elif upd_strat == 'cost':
            self.cost_upd = True
        else:
            raise StandardError, upd_strat

    cdef sort_moves(self):
        qsort(<void*>self.next_moves, self.nr_class, sizeof(Cont), cmp_contn)

    cdef bint accept(self, size_t parent, size_t move, double score):
        if self.seen_moves[parent][move] and not self.add_labels:
            return False
        self.seen_moves[parent][move] = True
        return True

    cdef State* add(self, size_t par_idx, double score, int cost) except NULL:
        cdef State* parent = self.parents[par_idx]
        assert par_idx < self.psize
        assert not self.is_full
        # TODO: Why's this broken?
        # If there are no more children coming, use the same state object instead
        # of cloning it
        #if parent.nr_kids > 1:
        copy_state(self.beam[self.bsize], parent, self.n_labels)
        #else:
        #    self.parents[par_idx] = self.beam[self.bsize]
        #    self.beam[self.bsize] = parent
        #    parent.nr_kids -= 1
        cdef State* ext = self.beam[self.bsize]
        ext.score = score
        ext.is_gold = ext.is_gold and cost == 0
        ext.cost += cost
        self.bsize += 1
        self.is_full = self.bsize >= self.k
        return ext

    cdef bint check_violation(self):
        cdef Violation violn
        cdef bint out_of_beam
        if self.bsize < self.k:
            return False
        if not self.beam[0].is_gold:
            if self.gold.score <= self.beam[0].score:
                out_of_beam = (not self.beam[self.k - 1].is_gold) and \
                        self.gold.score <= self.beam[self.k - 1].score
                violn = Violation()
                violn.set(self.beam[0], self.gold, out_of_beam)
                self.last_violn = violn
                if self.first_violn == None:
                    self.first_violn = violn
                    self.max_violn = violn
                    self.cost_violn = violn
                else:
                    if self.cost_violn.cost < violn.cost:
                        self.cost_violn = violn
                    if self.max_violn.delta <= violn.delta:
                        self.max_violn = violn
                        if self.cost_violn.cost == violn.cost:
                            self.cost_violn = violn
                return out_of_beam and self.early_upd
        return False

    cdef Violation pick_violation(self):
        assert self.first_violn is not None
        if self.early_upd:
            return self.first_violn
        elif self.max_upd:
            return self.max_violn
        elif self.late_upd:
            return self.last_violn
        elif self.cost_upd:
            return self.cost_violn
        else:
            raise StandardError, self.upd_strat

    cdef State* best_p(self) except NULL:
        if self.bsize != 0:
            return self.beam[0]
        else:
            raise StandardError

    cdef refresh(self):
        cdef size_t i
        for i in range(self.nr_class):
            for j in range(N_MOVES):
                self.seen_moves[i][j] = False

        for i in range(self.bsize):
            copy_state(self.parents[i], self.beam[i], self.n_labels)
        self.psize = self.bsize
        self.is_full = False
        self.bsize = 0

    def __dealloc__(self):
        for i in range(self.k):
            free_state(self.beam[i])
        for i in range(self.k):
            free_state(self.parents[i])
        for i in range(self.nr_class):
            free(self.seen_moves[i])
        free(self.next_moves)
        free(self.beam)
        free(self.parents)
        free_state(self.gold)


cdef class Violation:
    """
    A gold/prediction pair where the g.score < p.score
    """
    cdef size_t t
    cdef size_t* ghist
    cdef size_t* phist
    cdef double delta
    cdef int cost
    cdef bint out_of_beam

    def __cinit__(self):
        self.out_of_beam = False
        self.t = 0
        self.delta = 0.0
        self.cost = 0

    cdef int set(self, State* p, State* g, bint out_of_beam) except -1:
        self.delta = p.score - g.score
        self.cost = p.cost

        assert g.t == p.t, '%d vs %d' % (g.t, p.t)
        self.t = g.t
        self.ghist = <size_t*>malloc(self.t * sizeof(size_t))
        memcpy(self.ghist, g.history, self.t * sizeof(size_t))
        self.phist = <size_t*>malloc(self.t * sizeof(size_t))
        memcpy(self.phist, p.history, self.t * sizeof(size_t))
        self.out_of_beam = out_of_beam

    def __dealloc__(self):
        free(self.ghist)
        free(self.phist)



cdef class TransitionSystem:
    cdef bint allow_reattach
    cdef bint allow_reduce
    cdef size_t n_labels
    cdef object py_labels
    cdef int* _costs
    cdef size_t* labels
    cdef size_t* moves
    cdef size_t* l_classes
    cdef size_t* r_classes
    cdef list left_labels
    cdef list right_labels
    cdef size_t nr_class
    cdef size_t max_class
    cdef size_t s_id
    cdef size_t d_id
    cdef size_t l_start
    cdef size_t l_end
    cdef size_t r_start
    cdef size_t r_end

    def __cinit__(self, object labels, allow_reattach=False,
                  allow_reduce=False):
        self.n_labels = len(labels)
        self.py_labels = labels
        self.allow_reattach = allow_reattach
        self.allow_reduce = allow_reduce
        self.nr_class = 0
        max_classes = N_MOVES * len(labels)
        self.max_class = max_classes
        self._costs = <bint*>calloc(max_classes, sizeof(bint))
        self.labels = <size_t*>calloc(max_classes, sizeof(size_t))
        self.moves = <size_t*>calloc(max_classes, sizeof(size_t))
        self.l_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.r_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.s_id = 0
        self.d_id = 1
        self.l_start = 2
        self.l_end = 0
        self.r_start = 0
        self.r_end = 0

    def set_labels(self, left_labels, right_labels):
        self.left_labels = [self.py_labels[l] for l in sorted(left_labels)]
        self.right_labels = [self.py_labels[l] for l in sorted(right_labels)]
        self.labels[self.s_id] = 0
        self.labels[self.d_id] = 0
        self.moves[self.s_id] = <int>SHIFT
        self.moves[self.d_id] = <int>REDUCE
        clas = self.l_start
        for label in left_labels:
            self.moves[clas] = <int>LEFT
            self.labels[clas] = label
            self.l_classes[label] = clas
            clas += 1
        self.l_end = clas
        self.r_start = clas
        for label in right_labels:
            self.moves[clas] = <int>RIGHT
            self.labels[clas] = label
            self.r_classes[label] = clas
            clas += 1
        self.r_end = clas
        self.nr_class = clas
        return clas
        
    cdef int transition(self, size_t clas, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc, move, label
        move = self.moves[clas]
        label = self.labels[clas]
        s.history[s.t] = clas
        s.t += 1 
        if move == SHIFT:
            push_stack(s)
        elif move == REDUCE:
            if s.heads[s.top] == 0:
                assert self.allow_reduce
                assert s.second != 0
                assert s.second < s.top
                add_dep(s, s.second, s.top, s.guess_labels[s.top])
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
        else:
            raise StandardError(clas)
        if s.i == (s.n - 1):
            s.at_end_of_buffer = True
        if s.at_end_of_buffer and s.stack_len == 1:
            s.is_finished = True
  
    cdef int* get_costs(self, State* s, size_t* heads, size_t* labels) except NULL:
        cdef size_t i
        cdef int* costs = self._costs
        for i in range(self.nr_class):
            costs[i] = -1
        if s.stack_len == 1 and not s.at_end_of_buffer:
            costs[self.s_id] = 0
        if not s.at_end_of_buffer:
            costs[self.s_id] = self.s_cost(s, heads, labels)
            r_cost = self.r_cost(s, heads, labels)
            for i in range(self.r_start, self.r_end):
                if heads[s.i] == s.top and self.labels[i] != labels[s.i]:
                    costs[i] = r_cost + 1
                else:
                    costs[i] = r_cost
        if s.stack_len >= 2:
            costs[self.d_id] = self.d_cost(s, heads, labels)
            l_cost = self.l_cost(s, heads, labels)
            for i in range(self.l_start, self.l_end):
                if heads[s.top] == s.i and self.labels[i] != labels[s.top]:
                    costs[i] = l_cost + 1
                else:
                    costs[i] = l_cost
        return costs

    cdef int* get_valid(self, State* s):
        cdef size_t i
        cdef int* valid = self._costs
        for i in range(self.nr_class):
            valid[i] = -1
        if not s.at_end_of_buffer:
            valid[self.s_id] = 1
            if s.stack_len == 1:
                return valid
            else:
                for i in range(self.r_start, self.r_end):
                    valid[i] = 1
        if s.stack_len != 1:
            if s.heads[s.top] != 0:
                valid[self.d_id] = 1
            if self.allow_reattach or s.heads[s.top] == 0:
                for i in range(self.l_start, self.l_end):
                    valid[i] = 1
        if s.stack_len >= 3 and self.allow_reduce:
            valid[self.d_id] = 1
        return valid  

    cdef int break_tie(self, State* s, size_t* heads, size_t* labels) except -1:
        if s.stack_len == 1:
            return self.s_id
        elif not s.at_end_of_buffer and heads[s.i] == s.top:
            return self.r_classes[labels[s.i]]
        elif heads[s.top] == s.i and (self.allow_reattach or s.heads[s.top] == 0):
            return self.l_classes[labels[s.top]]
        elif self.d_cost(s, heads, labels) == 0:
            return self.d_id
        elif not s.at_end_of_buffer and self.s_cost(s, heads, labels) == 0:
            return self.s_id
        else:
            return self.nr_class + 1

    cdef int s_cost(self, State *s, size_t* heads, size_t* labels):
        cdef int cost = 0
        cdef size_t i, stack_i
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int r_cost(self, State *s, size_t* heads, size_t* labels):
        cdef int cost = 0
        cdef size_t i, buff_i, stack_i
        if heads[s.i] == s.top:
            return 0
        cost += has_head_in_buffer(s, s.i, heads)
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int d_cost(self, State *s, size_t* g_heads, size_t* g_labels):
        cdef int cost = 0
        if s.heads[s.top] == 0 and not self.allow_reduce:
            return -1
        if g_heads[s.top] == 0 and (s.stack_len == 2 or not self.allow_reattach):
            cost += 1
        cost += has_child_in_buffer(s, s.top, g_heads)
        if self.allow_reattach:
            cost += has_head_in_buffer(s, s.top, g_heads)
        return cost

    cdef int l_cost(self, State *s, size_t* heads, size_t* labels):
        cdef size_t buff_i, i
        cdef int cost = 0
        if s.heads[s.top] != 0 and not self.allow_reattach:
            return -1
        if heads[s.top] == s.i:
            return 0
        cost +=  has_head_in_buffer(s, s.top, heads)
        cost +=  has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach and heads[s.top] == s.heads[s.top]:
            cost += 1
        if self.allow_reduce and heads[s.top] == s.second:
            cost += 1
        return cost


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

def print_train_msg(n, n_corr, n_move, n_hit, n_miss, stats):
    pc = lambda a, b: '%.1f' % ((float(a) / b) * 100)
    move_acc = pc(n_corr, n_move)
    cache_use = pc(n_hit, n_hit + n_miss + 1e-100)
    msg = "#%d: Moves %d/%d=%s" % (n, n_corr, n_move, move_acc)
    if cache_use != 0:
        msg += '. Cache use %s' % cache_use
    if stats['early'] != 0:
        msg += '. Early %s' % pc(stats['early'], stats['sents'])
    msg += '. %.2f moves per sentence' % (float(stats['moves']) / stats['sents'])
    print msg

