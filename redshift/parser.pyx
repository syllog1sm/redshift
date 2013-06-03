# cython: profile=True
"""
MALT-style dependency parser
"""
cimport cython
import random
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset
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
from transitions cimport TransitionSystem, transition_to_str 
from beam cimport Beam, Violation
cimport features
from features cimport FeatureSet, ArcStandardFeatureSet

from io_parse import LABEL_STRS, STR_TO_LABEL

import index.hashes
cimport index.hashes

from learn.perceptron cimport Perceptron

from libc.stdint cimport uint64_t, int64_t
from libc.stdlib cimport qsort

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue


from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "sparsehash/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        uint64_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin()
        iterator end()
        uint64_t size()
        uint64_t max_size()
        bint empty()
        uint64_t bucket_count()
        uint64_t bucket_size(uint64_t i)
        uint64_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(uint64_t n)
        void rehash(uint64_t n)
        dense_hash_map()
        dense_hash_map(uint64_t n)
        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        uint64_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil


VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50
cdef double FOLLOW_ERR_PC = 0.90


DEBUG = False 
def set_debug(val):
    global DEBUG
    DEBUG = val

DEF USE_COLOURS = True

def red(string):
    if USE_COLOURS:
        return u'\033[91m%s\033[0m' % string
    else:
        return string


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

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  add_extra=True, label_set='MALT', feat_thresh=5,
                  allow_reattach=False, allow_reduce=False,
                  reuse_idx=False, beam_width=1, label_beam=True):
        model_dir = Path(model_dir)
        if not clean:
            params = dict([line.split() for line in model_dir.join('parser.cfg').open()])
            train_alg = params['train_alg']
            add_extra = True if params['add_extra'] == 'True' else False
            label_set = params['label_set']
            feat_thresh = int(params['feat_thresh'])
            allow_reattach = params['allow_reattach'] == 'True'
            allow_reduce = params['allow_reduce'] == 'True'
            l_labels = params['left_labels']
            r_labels = params['right_labels']
            beam_width = int(params['beam_width'])
            label_beam = params['label_beam'] == 'True'
        if allow_reattach and allow_reduce:
            print 'NM L+D'
        elif allow_reattach:
            print 'NM L'
        elif allow_reduce:
            print 'NM D'
        if beam_width >= 1:
            beam_settings = (beam_width, train_alg, label_beam)
            print 'Beam settings: k=%d; upd_strat=%s; label_beam=%s' % beam_settings
        self.model_dir = self.setup_model_dir(model_dir, clean)
        labels = io_parse.set_labels(label_set)
        if train_alg == 'standard':
            self.features = ArcStandardFeatureSet(len(labels), add_extra)
        else:
            print "Using ArcEager features"
            self.features = FeatureSet(len(labels), add_extra)
        self.add_extra = add_extra
        self.label_set = label_set
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        self.beam_width = beam_width
        self.label_beam = label_beam
        if clean == True:
            self.new_idx(self.model_dir, self.features.n)
        else:
            self.load_idx(self.model_dir, self.features.n)
        if self.train_alg == 'standard':
            raise StandardError
            #print "Using ArcStandard transitions"
            #self.moves = ArcStandard(labels)
        else:
            print "Using ArcEager transitions"
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
            sent = sents.s[i]
            for j in range(1, sent.length - 1):
                label = sent.parse.labels[j]
                if sent.parse.heads[j] > j:
                    seen_l_labels.add(label)
                else:
                    seen_r_labels.add(label)
        move_classes = self.moves.set_labels(seen_l_labels, seen_r_labels)
        self.guide.set_classes(range(move_classes))
        self.write_cfg(self.model_dir.join('parser.cfg'))
        if self.beam_width >= 1:
            self.guide.use_cache = True
        stats = defaultdict(int)
        indices = range(sents.length)
        for n in range(n_iter):
            random.shuffle(indices)
            # Group indices into minibatches of fixed size
            for i in indices:
                if DEBUG:
                    print ' '.join(sents.strings[i][0])
                if self.beam_width >= 1:
                    deltas = self.decode_beam(sents.s[i], self.beam_width, stats)
                    self.guide.batch_update(*deltas)
                else:
                    self.train_one(n, sents.s[i], sents.strings[i][0])
            print_train_msg(n, self.guide.n_corr, self.guide.total, self.guide.cache.n_hit,
                            self.guide.cache.n_miss, stats)
            self.guide.n_corr = 0
            self.guide.total = 0
            if n < 3:
                self.guide.reindex()
            #
        if self.feat_thresh > 1:
            self.guide.prune(self.feat_thresh)
        self.guide.finalize()

    cdef object decode_beam(self, Sentence* sent, size_t k, object stats):
        cdef size_t p_idx, i
        cdef Kernel* kernel
        cdef uint64_t* feats
        cdef int* costs
        cdef int cost
        cdef size_t* g_heads = sent.parse.heads
        cdef size_t* g_labels = sent.parse.labels
        cdef Violation violn
        cdef double* scores
        cdef bint cache_hit = False
        self.guide.cache.flush()
        cdef Beam beam = Beam(self.moves, k, sent.length, upd_strat=self.train_alg)
        stats['sents'] += 1
        beam_scores = <double**>malloc(beam.k * sizeof(double*))
        while not beam.gold.is_finished:
            self._advance_gold(beam.gold, sent, True)
            for p_idx in range(beam.bsize):
                kernel = beam.next_state(p_idx)
                beam.cost_next(p_idx, g_heads, g_labels)
                scores = self.guide.cache.lookup(sizeof(Kernel), kernel, &cache_hit)
                if not cache_hit:
                    feats = self.features.extract(sent, kernel)
                    self.guide.fill_scores(self.features.n, feats, scores)
                beam_scores[p_idx] = scores
            self.guide.cache.flush()
            beam.extend_states(beam_scores)
            beam.check_violation()
            if self.train_alg == 'early' and beam.violn != None:
                stats['early'] += 1
                break
            elif beam.beam[0].cost == 0:
                self.guide.n_corr += 1
        free(beam_scores)
        self.guide.total += beam.gold.t
        if beam.violn is not None:
            stats['moves'] += beam.violn.t
            counted = self._count_feats(sent, beam.violn.t, beam.violn.phist, beam.violn.ghist)
            return (counted, beam.violn.delta + 1)
        else:
            stats['moves'] += beam.gold.t
            return ({}, 0)

    cdef int _advance_gold(self, State* s, Sentence* sent, bint use_static) except -1:
        cdef:
            size_t oracle, i
            int* costs
            uint64_t* feats
            double* scores
            bint cache_hit
            double best_score
            double best_right
            size_t rlabel
        fill_kernel(s)
        scores = self.guide.cache.lookup(sizeof(s.kernel), <void*>&s.kernel, &cache_hit)
        if not cache_hit:
            feats = self.features.extract(sent, &s.kernel)
            self.guide.fill_scores(self.features.n, feats, scores)
        if use_static:
            oracle = self.moves.break_tie(s, sent.parse.heads, sent.parse.labels)
        else:
            costs = self.moves.get_costs(s, sent.parse.heads, sent.parse.labels)
            best_score = -1000000
            best_right = scores[self.moves.r_start]
            rlabel = self.moves.labels[self.moves.r_start]
            for i in range(self.moves.nr_class):
                if costs[i] == 0 and scores[i] > best_score:
                    oracle = i
                    best_score = scores[i]
                if self.moves.r_end > i > self.moves.r_start and scores[i] > best_right:
                    rlabel = self.moves.labels[i]
                    best_right = scores[i]
            assert best_score > -1000000
        s.guess_labels[s.i] = rlabel
        s.score += scores[oracle]
        self.moves.transition(oracle, s)

    cdef dict _count_feats(self, Sentence* sent, size_t t, size_t* phist, size_t* ghist):
        cdef size_t d, i, f
        cdef size_t n_feats = self.features.n
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(sent.length)
        cdef State* pred_state = init_state(sent.length)
        # Find where the states diverge
        for d in range(t):
            if ghist[d] == phist[d]:
                self.moves.transition(ghist[d], gold_state)
                self.moves.transition(phist[d], pred_state)
            else:
                break
        else:
            return {}
        cdef dict counts = {}
        for i in range(d, t):
            fill_kernel(gold_state)
            feats = self.features.extract(sent, &gold_state.kernel)
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
            fill_kernel(pred_state)
            feats = self.features.extract(sent, &pred_state.kernel)
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
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads

        cdef size_t n_feats = self.features.n
        cdef State* s = init_state(sent.length)
        cdef size_t move = 0
        cdef size_t label = 0
        cdef size_t _ = 0
        cdef bint online = self.train_alg == 'online'
        if DEBUG:
            print ' '.join(py_words)
        while not s.is_finished:
            fill_kernel(s)
            feats = self.features.extract(sent, &s.kernel)
            self.moves.fill_valid(s, valid)
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
            self.guide.n_corr += (gold == pred)
            self.guide.total += 1
        free_state(s)
        free(valid)

    def add_parses(self, Sentences sents, k=None, collect_stats=False):
        cdef:
            size_t i
        if k == None:
            k = self.beam_width
        self.guide.nr_class = self.moves.nr_class
        prune_freqs = {} if collect_stats else None
        for i in range(sents.length):
            if k == 0:
                self.parse(sents.s[i], sents.strings[i][0])
            else:
                self.beam_parse(sents.s[i], k, prune_freqs)
        if prune_freqs is not None:
            for k, v in sorted(prune_freqs.items()):
                print '%d\t%d' % (k, v)

    cdef int parse(self, Sentence* sent, words) except -1:
        cdef State* s
        cdef size_t n_preds = self.features.n
        cdef uint64_t* feats
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        while not s.is_finished:
            fill_kernel(s)
            feats = self.features.extract(sent, &s.kernel)
            try:
                self.moves.fill_valid(s, self.moves._costs)
                clas = self.predict(n_preds, feats, self.moves._costs,
                                    &s.guess_labels[s.i])
            except:
                print '%d stack, buffer=%d, len=%d' % (s.stack_len, s.i, s.n)
                raise
            sent.parse.moves[s.t] = clas
            self.moves.transition(clas, s)
        sent.parse.n_moves = s.t
        # No need to copy heads for root and start symbols

        cdef size_t root
        quot = index.hashes.encode_pos("``")
        comma = index.hashes.encode_pos(",")
        for i in range(1, sent.length - 1):
            assert s.heads[i] != 0, i
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]
            continue
            if s.r_valencies[i] == 0:
                root = i
                while s.labels[root] != 1:
                    if s.heads[root] > root:
                        break
                    if s.heads[root] == 0:
                        break
                    if get_r(s, s.heads[root]) != root:
                        break
                    elif sent.pos[root] == quot:
                        break
                    root = s.heads[root]
                else:
                    if sent.pos[i] != quot and sent.pos[i] != comma:
                        sent.parse.sbd[i] = 1
        free_state(s)
    
    cdef int beam_parse(self, Sentence* sent, size_t k, dict prune_freqs) except -1:
        cdef Beam beam = Beam(self.moves, k, sent.length, upd_strat=self.train_alg,
                              prune_freqs=prune_freqs)
        self.guide.cache.flush()
        cdef size_t p_idx, i
        cdef double* scores
        cdef bint cache_hit
        cdef Kernel* kernel
        cdef uint64_t* feats
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        while not beam.is_finished:
            for p_idx in range(beam.bsize):
                kernel = beam.next_state(p_idx)
                scores = self.guide.cache.lookup(sizeof(kernel[0]), <void*>kernel,
                                                 &cache_hit)
                if not cache_hit:
                    feats = self.features.extract(sent, kernel)
                    self.guide.fill_scores(self.features.n, feats, scores)
                beam_scores[p_idx] = scores
            beam.extend_states(beam_scores)
        sent.parse.n_moves = beam.t
        beam.fill_parse(sent.parse.moves, sent.parse.heads, sent.parse.labels,
                        sent.parse.sbd)
        free(beam_scores)

    cdef int predict(self, uint64_t n_preds, uint64_t* feats, int* valid,
                     size_t* rlabel) except -1:
        cdef:
            size_t i
            double score
            size_t clas, best_valid, best_right
            double* scores

        cdef size_t right_move = 0
        cdef double valid_score = -10000
        cdef double right_score = -10000
        scores = self.guide.scores
        self.guide.fill_scores(n_preds, feats, scores)
        seen_valid = False
        for clas in range(self.guide.nr_class):
            score = scores[clas]
            if valid[clas] == 0:
                if score > valid_score:
                    best_valid = clas
                    valid_score = score
                if not seen_valid:
                    seen_valid = True
            if self.moves.r_end > clas >= self.moves.r_start and score > right_score:
                best_right = clas
                right_score = score
        assert seen_valid 
        rlabel[0] = self.moves.labels[best_right]
        return best_valid

    def save(self):
        self.guide.save(str(self.model_dir.join('model')))

    def load(self):
        self.guide.load(str(self.model_dir.join('model')))

    def new_idx(self, model_dir, size_t n_predicates):
        index.hashes.init_word_idx(model_dir.join('words'))
        index.hashes.init_pos_idx(model_dir.join('pos'))

    def load_idx(self, model_dir, size_t n_predicates):
        model_dir = Path(model_dir)
        index.hashes.load_word_idx(model_dir.join('words'))
        index.hashes.load_pos_idx(model_dir.join('pos'))
   
    def write_cfg(self, loc):
        with loc.open('w') as cfg:
            cfg.write(u'model_dir\t%s\n' % self.model_dir)
            cfg.write(u'train_alg\t%s\n' % self.train_alg)
            cfg.write(u'add_extra\t%s\n' % self.add_extra)
            cfg.write(u'label_set\t%s\n' % self.label_set)
            cfg.write(u'feat_thresh\t%d\n' % self.feat_thresh)
            cfg.write(u'allow_reattach\t%s\n' % self.moves.allow_reattach)
            cfg.write(u'allow_reduce\t%s\n' % self.moves.allow_reduce)
            cfg.write(u'left_labels\t%s\n' % ','.join(self.moves.left_labels))
            cfg.write(u'right_labels\t%s\n' % ','.join(self.moves.right_labels))
            cfg.write(u'beam_width\t%d\n' % self.beam_width)
            cfg.write(u'label_beam\t%s\n' % self.label_beam)

    def __dealloc__(self):
        pass
      
    """
    def get_best_moves(self, Sentences sents, Sentences gold):
        "Get a list of move taken/oracle move pairs for output"
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
            s = init_state(n)
            sent_moves = []
            tokens = sents.strings[i][0]
            while not s.is_finished:
                costs = self.moves.get_costs(s, g_heads, g_labels)
                best_strs = []
                best_ids = set()
                for clas in range(self.moves.nr_class):
                    if costs[clas] == 0:
                        move = self.moves.moves[clas]
                        label = 0
                        if move not in best_ids:
                            best_strs.append(lmove_to_str(move, label, s.heads[s.top]))
                        best_ids.add(move)
                best_strs = ','.join(best_strs)
                best_id_str = ','.join(map(str, sorted(best_ids)))
                parse_class = sent.parse.moves[s.t]
                state_str = transition_to_str(s, self.moves.moves[parse_class],
                                              self.moves.labels[parse_class],
                                              tokens)
                parse_move = self.moves.moves[parse_class]
                parse_label = 0
                parse_move_str = lmove_to_str(parse_move, parse_label,
                                              s.heads[s.top])
                sent_moves.append((best_id_str, int(move),
                                  best_strs, parse_move_str,
                                  state_str))
                self.moves.transition(parse_class, s)
            free_state(s)
            best_moves.append((u' '.join(tokens), sent_moves))
        return best_moves
        """

def print_train_msg(n, n_corr, n_move, n_hit, n_miss, stats):
    pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
    move_acc = pc(n_corr, n_move)
    cache_use = pc(n_hit, n_hit + n_miss + 1e-100)
    msg = "#%d: Moves %d/%d=%s" % (n, n_corr, n_move, move_acc)
    if cache_use != 0:
        msg += '. Cache use %s' % cache_use
    if stats['early'] != 0:
        msg += '. Early %s' % pc(stats['early'], stats['sents'])
    if 'moves' in stats:
        msg += '. %.2f moves per sentence' % (float(stats['moves']) / stats['sents'])
    print msg

