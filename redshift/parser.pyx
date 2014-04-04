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

from _state cimport *
from sentence cimport Input, Sentence, Token
from transitions cimport Transition, transition, fill_valid, fill_costs
from transitions cimport get_nr_moves, fill_moves
from transitions cimport *
from beam cimport Beam
from tagger cimport Tagger
from tagger import write_tagger_config

from features.extractor cimport Extractor
import _parse_features
from _parse_features cimport *

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


def train(train_str, model_dir, n_iter=15, beam_width=8, train_tagger=True,
          feat_set='basic', feat_thresh=10,
          use_edit=False, use_break=False, use_filler=False):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    cdef list sents = [Input.from_conll(s) for s in
                       train_str.strip().split('\n\n') if s.strip()]
    left_labels, right_labels = get_labels(sents)
    Config.write(model_dir, beam_width=beam_width, features=feat_set,
                 feat_thresh=feat_thresh,
                 left_labels=left_labels, right_labels=right_labels,
                 use_edit=use_edit, use_break=use_break, use_filler=use_filler)
    write_tagger_config(model_dir, beam_width=4, features='basic', feat_thresh=feat_thresh)
    parser = Parser(model_dir)
    indices = list(range(len(sents)))
    cdef Input py_sent
    for n in range(n_iter):
        for i in indices:
            py_sent = sents[i]
            parser.tagger.train_sent(py_sent)
            parser.train_sent(py_sent)
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
    cdef Input sent
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            if sent.c_sent.tokens[j].head > j:
                left_labels.add(sent.c_sent.tokens[j].label)
            else:
                right_labels.add(sent.c_sent.tokens[j].label)
    return list(sorted(left_labels)), list(sorted(right_labels))


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def write(cls, model_dir, **kwargs):
        open(pjoin(model_dir, 'config.json'), 'w').write(json.dumps(kwargs))

    @classmethod
    def read(cls, model_dir):
        return cls(**json.load(open(pjoin(model_dir, 'config.json'))))


def get_templates(feats_str):
    match_feats = []
    templates = _parse_features.arc_hybrid
    if 'disfl' in feats_str:
        templates += _parse_features.disfl
        templates += _parse_features.new_disfl
        templates += _parse_features.suffix_disfl
        templates += _parse_features.extra_labels
        templates += _parse_features.clusters
        templates += _parse_features.edges
        match_feats = _parse_features.match_templates()
    elif 'clusters' in feats_str:
        templates += _parse_features.clusters
    if 'bitags' in feats_str:
        templates += _parse_features.pos_bigrams()
    return templates, match_feats


cdef class Parser:
    cdef object cfg
    cdef Extractor extractor
    cdef Perceptron guide
    cdef Tagger tagger
    cdef size_t beam_width
    cdef int feat_thresh
    cdef Transition* moves
    cdef uint64_t* _features
    cdef size_t* _context
    cdef size_t nr_moves

    def __cinit__(self, model_dir):
        assert os.path.exists(model_dir) and os.path.isdir(model_dir)
        self.cfg = Config.read(model_dir)
        self.extractor = Extractor(*get_templates(self.cfg.features))
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(_parse_features.context_size(), sizeof(size_t))

        self.feat_thresh = self.cfg.feat_thresh
        self.beam_width = self.cfg.beam_width

        if os.path.exists(pjoin(model_dir, 'labels')):
            index.hashes.load_label_idx(pjoin(model_dir, 'labels'))
        self.nr_moves = get_nr_moves(self.cfg.left_labels, self.cfg.right_labels,
                                     self.cfg.use_edit, self.cfg.use_break,
                                     self.cfg.use_filler)
        self.moves = <Transition*>calloc(self.nr_moves, sizeof(Transition))
        fill_moves(self.cfg.left_labels, self.cfg.right_labels, self.cfg.use_edit,
                   self.cfg.use_break, self.cfg.use_filler, self.moves)
        
        self.guide = Perceptron(self.nr_moves, pjoin(model_dir, 'model.gz'))
        if os.path.exists(pjoin(model_dir, 'model.gz')):
            self.guide.load(pjoin(model_dir, 'model.gz'), thresh=int(self.cfg.feat_thresh))
        if os.path.exists(pjoin(model_dir, 'pos')):
            index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        self.tagger = Tagger(model_dir)

    cpdef int parse(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t p_idx, i
        if self.tagger:
            self.tagger.tag(py_sent)
        cdef Beam beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves,
                              py_sent)

        self.guide.cache.flush()
        while not beam.is_finished:
            for i in range(beam.bsize):
                if not beam.beam[i].is_finished:
                    self._predict(beam.beam[i], beam.moves[i], sent.lattice)
                # The False flag tells it to allow non-gold predictions
                beam.enqueue(i, False)
            beam.extend()
        beam.fill_parse(sent.tokens)
        sent.score = beam.beam[0].score

    cdef int _predict(self, State* s, Transition* classes, Step* lattice) except -1:
        cdef bint cache_hit = False
        if s.is_finished:
            return 0
        fill_slots(s)
        scores = self.guide.cache.lookup(sizeof(SlotTokens), &s.slots, &cache_hit)
        if not cache_hit:
            fill_context(self._context, &s.slots, s.parse, lattice)
            self.extractor.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
        fill_valid(s, classes, self.nr_moves)
        for i in range(self.nr_moves):
            classes[i].score = scores[i]

    cdef int train_sent(self, Input py_sent) except -1:
        cdef size_t i
        cdef size_t nr_move = sent.n * 3
        cdef Transition[1000] g_hist
        cdef Transition[1000] p_hist
        cdef Sentence* sent = py_sent.c_sent
        
        cdef size_t* gold_tags = <size_t*>calloc(sent.n, sizeof(size_t))
        for i in range(sent.n):
            gold_tags[i] = sent.tokens[i].tag
        if self.tagger:
            self.tagger.tag(py_sent)
        g_beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
        p_beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
        cdef Token* gold_parse = sent.tokens
        cdef double delta = 0
        cdef double max_violn = -1
        cdef size_t pt = 0
        cdef size_t gt = 0
        cdef State* p
        cdef State* g
        cdef Transition* moves
        self.guide.cache.flush()
        words = py_sent.words
        #print words
        move_strs = ['?', 'S', 'L', 'R', 'E']
        while not p_beam.is_finished and not g_beam.is_finished:
            for i in range(p_beam.bsize):
                self._predict(p_beam.beam[i], p_beam.moves[i], sent.lattice)
                # Fill costs so we can see whether the prediction is gold-standard
                #print 'ignore',
                fill_costs(p_beam.beam[i], p_beam.moves[i], self.nr_moves, gold_parse)
                # The False flag tells it to allow non-gold predictions
                p_beam.enqueue(i, False)
            p_beam.extend()
            for i in range(g_beam.bsize):
                g = g_beam.beam[i]
                moves = g_beam.moves[i]
                self._predict(g, moves, sent.lattice)
                # Constrain this beam to only gold candidates
                #print g.stack_len
                #print words[g.second], words[g.top], '|', words[g.i]
                #print sent.tokens[g.second].is_edit, sent.tokens[g.top].is_edit, sent.tokens[g.i].is_edit
                fill_costs(g, moves, self.nr_moves, gold_parse)
                g_beam.enqueue(i, True)
            g_beam.extend()
            g = g_beam.beam[0]; p = p_beam.beam[0] 
            #print 'Move taken: ', move_strs[g.history[g.m-1].move]
            delta = p.score - g.score
            if delta >= max_violn and p.cost >= 1:
                max_violn = delta
                pt = p.m
                gt = g.m
                memcpy(p_hist, p.history, pt * sizeof(Transition))
                memcpy(g_hist, g.history, gt * sizeof(Transition))
        if max_violn >= 0:
            counted = self._count_feats(sent, pt, gt, p_hist, g_hist)
            self.guide.batch_update(counted)
            # TODO: We should tick the epoch here if max_violn == 0, right?
        #else:
        #    self.guide.now += 1
        for i in range(sent.n):
            sent.tokens[i].tag = gold_tags[i]
        free(gold_tags)

    cdef dict _count_feats(self, Sentence* sent, size_t pt, size_t gt,
                           Transition* phist, Transition* ghist):
        cdef size_t d, i, f
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(sent)
        cdef State* pred_state = init_state(sent)
        # Find where the states diverge
        cdef dict counts = {}
        for clas in range(self.nr_moves):
            counts[clas] = {}
        cdef bint seen_diff = False
        g_inc = 1.0
        p_inc = -1.0
        for i in range(max((pt, gt))):
            self.guide.total += 1
            if not seen_diff and ghist[i].clas == phist[i].clas:
                self.guide.n_corr += 1
                transition(&ghist[i], gold_state)
                transition(&phist[i], pred_state)
                continue
            seen_diff = True
            if i < gt:
                self._inc_feats(counts[ghist[i].clas], gold_state, sent.lattice, g_inc)
                transition(&ghist[i], gold_state)
            if i < pt:
                self._inc_feats(counts[phist[i].clas], pred_state, sent.lattice, p_inc)
                transition(&phist[i], pred_state)
        free_state(gold_state)
        free_state(pred_state)
        return counts

    cdef int _inc_feats(self, dict counts, State* s, Step* lattice, double inc) except -1:
        fill_slots(s)
        fill_context(self._context, &s.slots, s.parse, lattice)
        self.extractor.extract(self._features, self._context)
 
        cdef size_t f = 0
        while self._features[f] != 0:
            if self._features[f] not in counts:
                counts[self._features[f]] = 0
            counts[self._features[f]] += inc
            f += 1
