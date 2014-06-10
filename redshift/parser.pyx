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


def train(sents, model_dir, n_iter=15, beam_width=8,
          feat_set='basic', feat_thresh=10, use_break=False):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    shift_classes, lattice_width, left_labels, right_labels, dfl_labels = get_labels(sents)
    Config.write(model_dir, 'config', beam_width=beam_width, features=feat_set,
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
        for i in indices:
            py_sent = sents[i]
            # TODO: Should this be sensitive to whether we've hit max trainign
            # iters for tagger?
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


def train_nbest(sents, nbests, model_dir, n_iter=15, beam_width=8,
                feat_set='basic', feat_thresh=10,
                use_break=False):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    gold_sents = list(sents)
    cdef Input sent
    for nbest in nbests:
        for sent in nbest:
            if sent.wer == 0:
                gold_sents.append(sent)
    lattice_classes, lattice_width, left_labels, right_labels, dfl_labels = get_labels(gold_sents)
    Config.write(model_dir, 'config', beam_width=beam_width, features=feat_set,
                 feat_thresh=feat_thresh,
                 lattice_classes=lattice_classes,
                 lattice_width=lattice_width,
                 left_labels=left_labels, right_labels=right_labels,
                 dfl_labels=dfl_labels, use_break=use_break)
    Config.write(model_dir, 'tagger', beam_width=4, features='basic',
                 feat_thresh=5)
    parser = Parser(model_dir)
    indices = list(range(len(sents)))
    for n in range(n_iter):
        for i in indices:
            parser.tagger.train_sent(sents[i])
            parser.train_nbest(nbests[i])
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
    nr_lattice_classes = 1
    assert lattice_width >= nr_lattice_classes
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
    if 'disfl' in feats_str:
        templates += _parse_features.disfl
        templates += _parse_features.new_disfl
        templates += _parse_features.suffix_disfl
        templates += _parse_features.extra_labels
        templates += _parse_features.clusters
        templates += _parse_features.edges
        templates += _parse_features.prev_next
        templates += _parse_features.string_probs
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
        self.cfg = Config.read(model_dir, 'config')
        self.extractor = Extractor(*get_templates(self.cfg.features))
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(_parse_features.context_size(), sizeof(size_t))

        self.feat_thresh = self.cfg.feat_thresh
        self.beam_width = self.cfg.beam_width
 
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
        self.guide = Perceptron(self.nr_moves - self.cfg.lattice_width + 1,
                                pjoin(model_dir, 'model.gz'))
        if os.path.exists(pjoin(model_dir, 'model.gz')):
            self.guide.load(pjoin(model_dir, 'model.gz'), thresh=int(self.cfg.feat_thresh))
        if os.path.exists(pjoin(model_dir, 'pos')):
            index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        self.tagger = Tagger(model_dir)

    cpdef int parse(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t p_idx, i
        cdef Beam beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves,
                              py_sent)
        self.guide.cache.flush()
        while not beam.is_finished:
            for i in range(beam.bsize):
                if not is_final(beam.beam[i]):
                    fill_valid(beam.beam[i], sent.lattice, beam.moves[i], self.nr_moves) 
                    self.tagger.tag_word(beam.beam[i].parse, beam.beam[i].i+1,
                                         sent.lattice, sent.n)
                    self.tagger.tag_word(beam.beam[i].parse, beam.beam[i].i+2,
                                         sent.lattice, sent.n)
                    self._score_classes(beam.beam[i], beam.moves[i])
            beam.extend()
        cdef State* s = beam.beam[0]
        for i in range(sent.n):
            tag = decode_pos(s.parse[i].tag)
            word = get_str(<size_t>s.parse[i].word)
            self.tagger.tag_word(s.parse, i, sent.lattice, sent.n)
            new_tag = decode_pos(s.parse[i].tag)
        beam.fill_parse(sent.tokens)
        py_sent.segment()
        sent.score = beam.score

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

    cdef int _prepare_state(self, State* s, Token* tokens, Step* lattice) except -1:
        for i in range(s.n):
            s.parse[i].tag = tokens[i].tag
            s.parse[i].word = lattice[i].nodes[0]

    cdef int train_sent(self, Input py_sent) except -1:
        cdef size_t i
        cdef State* s
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t* gold_tags = <size_t*>calloc(sent.n, sizeof(size_t))
        for i in range(sent.n):
            gold_tags[i] = sent.tokens[i].tag
        self.tagger.tag(py_sent)
        cdef Token* gold_parse = sent.tokens
        self.guide.cache.flush()
        p_beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
        for i in range(self.beam_width):
            self._prepare_state(p_beam.beam[i], sent.tokens, sent.lattice)
        while not p_beam.is_finished:
            for i in range(p_beam.bsize):
                s = p_beam.beam[i]
                if not is_final(s):
                    fill_valid(s, sent.lattice, p_beam.moves[i], self.nr_moves) 
                    self.tagger.tag_word(s.parse, s.i+1, sent.lattice, sent.n)
                    self.tagger.tag_word(s.parse, s.i+2, sent.lattice, sent.n)
                    self._score_classes(s, p_beam.moves[i])
                    # Fill costs so we can see whether the prediction is gold-standard
                    fill_costs(s, sent.lattice, p_beam.moves[i], self.nr_moves,
                               gold_parse)
            p_beam.extend()
        if p_beam.beam[0].cost == 0:
            self.guide.now += 1
            self.guide.total += 1
            self.guide.n_corr += 1
            for i in range(sent.n):
                sent.tokens[i].tag = gold_tags[i]
            free(gold_tags)
            return 0
        g_beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
        for i in range(self.beam_width):
            self._prepare_state(g_beam.beam[i], sent.tokens, sent.lattice)
  
        while not g_beam.is_finished:
            for i in range(g_beam.bsize):
                s = g_beam.beam[i]
                if not is_final(s):
                    self.tagger.tag_word(s.parse, s.i+1, sent.lattice, sent.n)
                    self.tagger.tag_word(s.parse, s.i+2, sent.lattice, sent.n)
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
        if v < 0:
            return {}
        cdef Transition* phist
        cdef Transition* ghist
        cdef size_t pt, gt
        if v >= g_beam.t:
            ghist = g_beam.history[g_beam.t - 1]
            gt = g_beam.lengths[g_beam.t - 1]
        else:
            ghist = g_beam.history[v]
            gt = g_beam.lengths[v]
        if v >= p_beam.t:
            phist = p_beam.history[p_beam.t - 1]
            pt = p_beam.lengths[p_beam.t - 1]
        else:
            phist = p_beam.history[v]
            pt = p_beam.lengths[v]

        cdef size_t d, i, f
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(gsent.n)
        cdef State* pred_state = init_state(psent.n)
        self._prepare_state(gold_state, gsent.tokens, gsent.lattice)
        self._prepare_state(pred_state, psent.tokens, psent.lattice)
        cdef dict counts = {}
        for clas in range(self.nr_moves):
            counts[clas] = {}
        cdef bint seen_diff = False
        cdef Token* gword
        cdef Token* pword
        for i in range(max((pt, gt))):
            self.tagger.tag_word(gold_state.parse, gold_state.i+1, gsent.lattice, gsent.n)
            self.tagger.tag_word(gold_state.parse, gold_state.i+2, gsent.lattice, gsent.n)
            self.tagger.tag_word(pred_state.parse, pred_state.i+1, psent.lattice, psent.n)
            self.tagger.tag_word(pred_state.parse, pred_state.i+2, psent.lattice, psent.n)
            # Find where the states diverge
            gword = &gsent.tokens[gold_state.i]
            pword = &psent.tokens[pred_state.i]
            if not seen_diff and gword.word == pword.word and ghist[i].clas == phist[i].clas:
                transition(&ghist[i], gold_state, gsent.lattice)
                transition(&phist[i], pred_state, psent.lattice)
                continue
            seen_diff = True
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



"""
    cdef int train_nbest(self, object nbest) except -1:
        cdef Input py_sent
        cdef Sentence* psent
        cdef Sentence* gsent
        cdef Sentence* sent
        cdef size_t i
        self.guide.cache.flush()
        # Identify best-scoring candidate, so we can search for max. violation
        # update within it.
        cdef Beam p_beam = None
        cdef Beam g_beam = None
        for py_sent in nbest:
            sent = py_sent.c_sent
            beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
            while not beam.is_finished:
                for i in range(beam.bsize):
                    s = beam.beam[i]
                    if not is_final(s):
                        fill_valid(s, sent.lattice, beam.moves[i], self.nr_moves) 
                        self.tagger.tag_word(s.parse, s.i+1, sent.lattice, sent.n)
                        self.tagger.tag_word(s.parse, s.i+2, sent.lattice, sent.n)
                        self._score_classes(s, beam.moves[i])
                        # Fill costs so we can see whether the prediction is gold-standard
                        if py_sent.wer == 0:
                            fill_costs(s, sent.lattice, beam.moves[i], self.nr_moves,
                                       sent.tokens)
                beam.extend()
            if p_beam is None or beam.score > p_beam.score:
                p_beam = beam
                psent = sent
            
            if py_sent.wer != 0:
                continue
            beam = Beam(self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
            while not beam.is_finished:
                for i in range(beam.bsize):
                    s = beam.beam[i]
                    if not is_final(s):
                        fill_valid(s, sent.lattice, beam.moves[i], self.nr_moves) 
                        fill_costs(s, sent.lattice, beam.moves[i], self.nr_moves,
                                   sent.tokens)
                        self.tagger.tag_word(s.parse, s.i+1, sent.lattice, sent.n)
                        self.tagger.tag_word(s.parse, s.i+2, sent.lattice, sent.n)
                        for j in range(self.nr_moves):
                            if beam.moves[i][j].cost != 0:
                                beam.moves[i][j].is_valid = False
                        self._score_classes(s, beam.moves[i])
                beam.extend()
            if g_beam is None or beam.score > g_beam.score:
                g_beam = beam
                gsent = sent
        assert p_beam is not None and g_beam is not None
        self.guide.total += 1
        if p_beam.beam[0].cost > 0:
            counts = self._count_feats(psent, gsent, p_beam, g_beam)
            self.guide.batch_update(counts)
        else:
            self.guide.n_corr += 1
            self.guide.now += 1
""" 
