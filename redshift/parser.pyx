"""
MALT-style dependency parser
"""
cimport cython
import random
import os.path
from os.path import join as pjoin
import shutil
import json

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset
from libcpp.vector cimport vector
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc

from _state cimport *
from sentence cimport Input, Sentence, Token, Step
from transitions cimport Transition, transition, fill_valid, fill_costs
from transitions cimport get_nr_moves, fill_moves
from transitions cimport *
from index.lexicon cimport get_str
from index.hashes import decode_pos
from beam cimport Beam, get_violation
from tagger cimport Tagger
from util import Config

from features.extractor cimport Extractor

import _lattice_features as _parse_features
from redshift._lattice_features cimport *

import index.hashes
cimport index.hashes

from learn.perceptron cimport Perceptron

from libc.stdint cimport uint64_t, int64_t


VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50

DEBUG = False 
def set_debug(val):
    global DEBUG
    DEBUG = val


def train(sents, model_dir, n_iter=15, beam_width=8, beam_factor=0.5,
          feat_set='basic', feat_thresh=10, use_break=False):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    shift_classes, lattice_width, left_labels, right_labels, dfl_labels = get_labels(sents)
    Config.write(model_dir, 'config', beam_width=beam_width, beam_factor=beam_factor,
                 features=feat_set,
                 feat_thresh=feat_thresh,
                 shift_classes=shift_classes,
                 lattice_width=lattice_width,
                 left_labels=left_labels, right_labels=right_labels,
                 dfl_labels=dfl_labels, use_break=use_break)
    Config.write(model_dir, 'tagger', beam_width=1, features='basic',
                 feat_thresh=5)
    parser = Parser(model_dir)
    indices = list(range(len(sents)))
    cdef Input py_sent
    for n in range(n_iter):
        parser.total_bsize = 0
        parser.beam_iters = 0
        parser.total_delta = 0
        parser.total_v = 0
        for i in indices:
            py_sent = sents[i]
            # TODO: Should this be sensitive to whether we've hit max trainign
            # iters for tagger?
            parser.tagger.train_sent(py_sent)
            parser.train_sent(py_sent)
        print 'Avg beam: %.2f' % parser.avg_bsize
        print 'Avg delta: %.1f' % parser.avg_delta
        print 'Avg V: %.1f' % parser.avg_v
        parser.guide.end_train_iter(n, feat_thresh)
        parser.tagger.guide.end_train_iter(n, feat_thresh)
        random.shuffle(indices)
    parser.guide.end_training(pjoin(model_dir, 'model.gz'))
    parser.tagger.guide.end_training(pjoin(model_dir, 'tagger.gz'))
    index.hashes.save_pos_idx(pjoin(model_dir, 'pos'))
    index.hashes.save_label_idx(pjoin(model_dir, 'labels'))
    return parser


def get_labels(sents):
    left_labels = set()
    right_labels = set()
    dfl_labels = set()
    cdef Input sent
    lattice_width = 0
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            if sent.c_sent.tokens[j].is_edit:
                dfl_labels.add(sent.c_sent.tokens[j].label)
            elif sent.c_sent.tokens[j].head > j:
                left_labels.add(sent.c_sent.tokens[j].label)
            else:
                right_labels.add(sent.c_sent.tokens[j].label)
            if sent.c_sent.lattice[j].n > lattice_width:
                lattice_width = sent.c_sent.lattice[j].n
    nr_lattice_classes = min(1, lattice_width)
    output = (
        nr_lattice_classes,
        lattice_width,
        list(sorted(left_labels)),
        list(sorted(right_labels)),
        list(sorted(dfl_labels))
    )
    return output


def get_templates(feats_str):
    match_feats = []
    templates = _parse_features.arc_hybrid
    templates += _parse_features.disfl
    templates += _parse_features.clusters
    templates += _parse_features.edges
    templates += _parse_features.string_probs
    if 'ngram' in feats_str:
        templates += _parse_features.bigrams
        templates += _parse_features.trigrams
    return templates, match_feats


cdef class Parser:
    cdef object cfg
    cdef Extractor extractor
    cdef Perceptron guide
    cdef Tagger tagger
    cdef size_t beam_width
    cdef double beta
    cdef int feat_thresh
    cdef Transition* moves
    cdef uint64_t* _features
    cdef size_t* _context
    cdef size_t nr_moves
    cdef size_t total_bsize
    cdef double total_delta
    cdef size_t total_v
    cdef size_t beam_iters

    def __cinit__(self, model_dir):
        assert os.path.exists(model_dir) and os.path.isdir(model_dir)
        self.cfg = Config.read(model_dir, 'config')
        self.extractor = Extractor(*get_templates(self.cfg.features))
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(_parse_features.context_size(), sizeof(size_t))
        
        self.total_bsize = 0
        self.total_delta = 0
        self.total_v = 0
        self.beam_iters = 0
        self.feat_thresh = self.cfg.feat_thresh
        self.beam_width = self.cfg.beam_width
        self.beta = self.cfg.beam_factor
 
        if os.path.exists(pjoin(model_dir, 'labels')):
            index.hashes.load_label_idx(pjoin(model_dir, 'labels'))
        self.nr_moves = get_nr_moves(self.cfg.shift_classes, self.cfg.lattice_width,
                                     self.cfg.left_labels,
                                     self.cfg.right_labels,
                                     self.cfg.dfl_labels,
                                     self.cfg.use_break)
        self.moves = <Transition*>calloc(self.nr_moves, sizeof(Transition))
        fill_moves(self.cfg.shift_classes, self.cfg.lattice_width,
                   self.cfg.left_labels, self.cfg.right_labels, self.cfg.dfl_labels,
                   self.cfg.use_break, self.moves)
        # One class for the distinct shift classes each, and 1 between the rest of the shifts
        nr_class = (self.nr_moves - self.cfg.lattice_width) + self.cfg.shift_classes + 1 
        self.guide = Perceptron(nr_class, pjoin(model_dir, 'model.gz'))
        if os.path.exists(pjoin(model_dir, 'model.gz')):
            self.guide.load(pjoin(model_dir, 'model.gz'), thresh=int(self.cfg.feat_thresh))
        if os.path.exists(pjoin(model_dir, 'pos')):
            index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        self.tagger = Tagger(model_dir)

    property avg_bsize:
        def __get__(self):
            return float(self.total_bsize) / self.beam_iters

    property avg_delta:
        def __get__(self):
            return float(self.total_delta) / self.beam_iters

    property avg_v:
        def __get__(self):
            return float(self.total_v) / (self.guide.total - self.guide.n_corr)

    cpdef int parse(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t p_idx, i
        cdef Beam beam = Beam(self.beta, self.beam_width, <size_t>self.moves,
                              self.nr_moves, py_sent)
        self.tagger.tag(py_sent)
        self.guide.cache.flush()
        cdef State* s
        while not beam.is_finished:
            for i in range(beam.bsize):
                s = beam.beam[i]
                if not is_final(s):
                    fill_valid(s, sent.lattice, beam.moves[i], self.nr_moves) 
                    self.tagger.tag_word(s.parse, s.i-1, sent.lattice, sent.n)
                    self.tagger.tag_word(s.parse, s.i, sent.lattice, sent.n)
                    self._score_classes(beam.beam[i], beam.moves[i])
            beam.extend()
        beam.fill_parse(sent.tokens)
        py_sent.segment()
        sent.score = beam.beam[0].score

    cdef int _score_classes(self, State* s, Transition* classes) except -1:
        assert not is_final(s)
        cdef bint cache_hit = False
        fill_slots(s)
        scores = self.guide.cache.lookup(sizeof(SlotTokens), &s.slots, &cache_hit)
        if not cache_hit:
            fill_context(self._context, &s.slots)
            self.extractor.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
        for i in range(self.nr_moves):
            classes[i].score = s.score + scores[classes[i].clas]
        return 0

    cdef int train_sent(self, Input py_sent) except -1:
        cdef size_t i
        cdef State* s
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t* gold_tags = <size_t*>calloc(sent.n, sizeof(size_t))
        cdef Token* gold_parse = sent.tokens
        for i in range(sent.n):
            gold_tags[i] = sent.tokens[i].tag
        self.tagger.tag(py_sent)
        self.guide.cache.flush()
        p_beam = Beam(self.beta, self.beam_width, <size_t>self.moves, self.nr_moves,
                      py_sent)
        while not p_beam.is_finished:
            for i in range(p_beam.bsize):
                s = p_beam.beam[i]
                if not is_final(s):
                    fill_valid(s, sent.lattice, p_beam.moves[i], self.nr_moves) 
                    self.tagger.tag_word(s.parse, s.i-1, sent.lattice, sent.n)
                    self.tagger.tag_word(s.parse, s.i, sent.lattice, sent.n)
                    self._score_classes(s, p_beam.moves[i])
                    # Fill costs so we can see whether the prediction is gold-standard
                    if s.cost == 0:
                        fill_costs(s, sent.lattice, p_beam.moves[i], self.nr_moves,
                                   gold_parse)
            self.total_bsize += p_beam.bsize
            self.beam_iters += 1
            p_beam.extend()
        if p_beam.beam[0].cost == 0:
            self.guide.now += 1
            self.guide.total += 1
            self.guide.n_corr += 1
            for i in range(sent.n):
                sent.tokens[i].tag = gold_tags[i]
            free(gold_tags)
            return 0
        g_beam = Beam(self.beta, self.beam_width, <size_t>self.moves, self.nr_moves,
                      py_sent)
  
        while not g_beam.is_finished:
            for i in range(g_beam.bsize):
                s = g_beam.beam[i]
                if not is_final(s):
                    self.tagger.tag_word(s.parse, s.i-1, sent.lattice, sent.n)
                    self.tagger.tag_word(s.parse, s.i, sent.lattice, sent.n)
                    fill_valid(s, sent.lattice, g_beam.moves[i], self.nr_moves) 
                    fill_costs(s, sent.lattice, g_beam.moves[i], self.nr_moves,
                               gold_parse)
                    for j in range(self.nr_moves):
                        if g_beam.moves[i][j].cost != 0:
                            g_beam.moves[i][j].is_valid = False
                    self._score_classes(s, g_beam.moves[i])

            g_beam.extend()
        self.guide.total += 1
        counts = self._count_feats(sent, sent, p_beam, g_beam)
        self.guide.batch_update(counts)
        for i in range(sent.n):
            sent.tokens[i].tag = gold_tags[i]
        free(gold_tags)
   
    cdef dict _count_feats(self, Sentence* psent, Sentence* gsent,
                           Beam p_beam, Beam g_beam):
        # TODO: Improve this...
        cdef int v = get_violation(p_beam, g_beam)
        if v < 1:
            return {}
        self.total_v += v
        cdef size_t pt, gt
        cdef double ps, gs
        cdef Transition* ghist
        cdef Transition* phist
        phist = p_beam.hist_at(v)
        ghist = g_beam.hist_at(v)
        pt = p_beam.length_at(v)
        gt = g_beam.length_at(v)
        ps = p_beam.score_at(v)
        gs = g_beam.score_at(v)
        
        self.total_delta += ps - gs
        if (ps - gs) < -1:
            print pt, gt
            print ps, gs
            print v
            raise StandardError
        cdef size_t d, i, f
        cdef uint64_t* feats
        for i in range(gt):
            assert ghist[i].move != 0
        for i in range(pt):
            assert phist[i].move != 0
        cdef size_t clas
        cdef State* gold_state = init_state(gsent.n)
        cdef State* pred_state = init_state(psent.n)
        cdef dict counts = {}
        for clas in range(self.nr_moves):
            counts[clas] = {}
        for i in range(max(pt, gt) + 1):
            self.tagger.tag_word(gold_state.parse, gold_state.i-1, gsent.lattice, gsent.n)
            self.tagger.tag_word(gold_state.parse, gold_state.i, gsent.lattice, gsent.n)
            self.tagger.tag_word(pred_state.parse, pred_state.i-1, psent.lattice, psent.n)
            self.tagger.tag_word(pred_state.parse, pred_state.i, psent.lattice, psent.n)
            if i < gt:
                fill_slots(gold_state)
                fill_context(self._context, &gold_state.slots)
                self.extractor.extract(self._features, self._context)
                self.extractor.count(counts[ghist[i].clas], self._features, 1.0)
                transition(&ghist[i], gold_state, gsent.lattice)
            if i < pt:
                fill_slots(pred_state)
                fill_context(self._context, &pred_state.slots)
                self.extractor.extract(self._features, self._context)
                self.extractor.count(counts[phist[i].clas], self._features, -1.0)
                transition(&phist[i], pred_state, psent.lattice)
        free_state(gold_state)
        free_state(pred_state)
        return counts
