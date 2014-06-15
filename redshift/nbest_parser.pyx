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


def get_labels(golds, nbests):
    sents = list(golds)
    cdef Input sent
    for nbest in nbests:
        for prob, sent in nbest:
            if sent.wer == 0:
                sents.append(sent)

    left_labels = set()
    right_labels = set()
    dfl_labels = set()
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            if sent.c_sent.tokens[j].is_edit:
                dfl_labels.add(sent.c_sent.tokens[j].label)
            elif sent.c_sent.tokens[j].head > j:
                left_labels.add(sent.c_sent.tokens[j].label)
            else:
                right_labels.add(sent.c_sent.tokens[j].label)
    output = (
        list(sorted(left_labels)),
        list(sorted(right_labels)),
        list(sorted(dfl_labels))
    )
    return output


def train(sents, nbests, model_dir, n_iter=15, beam_width=8,
          feat_set='basic', feat_thresh=10, use_break=False):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    left_labels, right_labels, dfl_labels = get_labels(sents, nbests)
    Config.write(model_dir, 'config', beam_width=beam_width, features=feat_set,
                 feat_thresh=feat_thresh,
                 shift_classes=0,
                 lattice_width=1,
                 left_labels=left_labels, right_labels=right_labels,
                 dfl_labels=dfl_labels, use_break=use_break)
    Config.write(model_dir, 'tagger', beam_width=4, features='basic',
                 feat_thresh=5)
    parser = NBestParser(model_dir)
    indices = list(range(len(sents)))
    for n in range(n_iter):
        for i in indices:
            parser.tagger.train_sent(sents[i])
            parser.train_sent(nbests[i])
        parser.guide.end_train_iter(n, feat_thresh)
        parser.tagger.guide.end_train_iter(n, feat_thresh)
        random.shuffle(indices)
    parser.guide.end_training(pjoin(model_dir, 'model.gz'))
    parser.tagger.guide.end_training(pjoin(model_dir, 'tagger.gz'))
    index.hashes.save_pos_idx(pjoin(model_dir, 'pos'))
    index.hashes.save_label_idx(pjoin(model_dir, 'labels'))
    return parser


def mix_weights(probs, scores, mix_weight=1.0):
    shift = abs(min(scores)) + 1
    scores = [score + shift for score in scores]
    mean = sum(scores) / len(scores)
    mixed = [score * prob for score, prob in zip(scores, probs)]
    return mixed


cdef class NBestParser:
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
        self.nr_moves = get_nr_moves(0, 1,
                                     self.cfg.left_labels,
                                     self.cfg.right_labels,
                                     self.cfg.dfl_labels,
                                     self.cfg.use_break)
        self.moves = <Transition*>calloc(self.nr_moves, sizeof(Transition))
        fill_moves(0, 1, self.cfg.left_labels, self.cfg.right_labels,
                   self.cfg.dfl_labels, self.cfg.use_break, self.moves)
        self.guide = Perceptron(self.nr_moves, pjoin(model_dir, 'model.gz'))
        if os.path.exists(pjoin(model_dir, 'model.gz')):
            self.guide.load(pjoin(model_dir, 'model.gz'), thresh=int(self.cfg.feat_thresh))
        if os.path.exists(pjoin(model_dir, 'pos')):
            index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        self.tagger = Tagger(model_dir)

    def parse_nbest(self, candidates, mix_weight=1.0):
        sents = []
        scores = []
        probs = []
        cdef Input py_sent
        for prob, tokens in candidates:
            py_sent = Input.from_strings(tokens)
            beam = self.search(py_sent, False, False)
            beam.fill_parse(py_sent.c_sent.tokens)
            scores.append(beam.score)
            probs.append(prob)
            sents.append(py_sent)
        weights = mix_weights(probs, scores, mix_weight=mix_weight)
        for weight, py_sent in zip(weights, sents):
            py_sent.c_sent.score = weight
        return sents

    cpdef int parse(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t p_idx, i
        self.guide.cache.flush()
        cdef Beam beam = self.search(py_sent, False, False)
        beam.fill_parse(py_sent.c_sent.tokens)
        sent.score = beam.score

    cdef int train_sent(self, object nbest) except -1:
        cdef Input py_sent
        cdef size_t i
        self.guide.total += 1
        self.guide.cache.flush()
        # Identify best-scoring candidate, so we can search for max. violation
        # update within it.
        cdef Beam p_beam = self._nbest_search(nbest, False)
        cdef Beam g_beam = self._nbest_search(nbest, True)

        if p_beam.cost > 0:
            counts = self._count_feats(p_beam, g_beam)
            self.guide.batch_update(counts)
        else:
            self.guide.n_corr += 1
            self.guide.now += 1

    cdef Beam search(self, Input py_sent, bint force_gold, bint set_costs):
        self.tagger.tag(py_sent)
        cdef Sentence* sent = py_sent.c_sent
        cdef Beam b = Beam(0.0, self.beam_width, <size_t>self.moves, self.nr_moves, py_sent)
        cdef size_t i, w
        for i in range(self.beam_width):
            for w in range(sent.n):
                b.beam[i].parse[w].tag = sent.tokens[w].tag
                
        cdef State* s
        while not b.is_finished:
            for i in range(b.bsize):
                s = b.beam[i]
                if not is_final(s):
                    self._validate_moves(b.moves[i], s, sent, force_gold, set_costs)
                    self._score_classes(b.moves[i], s)
            b.extend()
        return b

    cdef int _validate_moves(self, Transition* moves, State* s, Sentence* sent,
                             bint force_gold, bint add_costs) except -1:
        fill_valid(s, sent.lattice, moves, self.nr_moves) 
        if add_costs or force_gold:
            fill_costs(s, sent.lattice, moves, self.nr_moves, sent.tokens)
        if force_gold:
            for i in range(self.nr_moves):
                if moves[i].cost != 0:
                    moves[i].is_valid = False

    cdef int _score_classes(self, Transition* classes, State* s) except -1:
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

    cdef Beam _nbest_search(self, list nbest, bint force_gold):
        cdef Input py_sent
        beams = []
        scores = []
        probs = []
        for prob, py_sent in nbest:
            if force_gold and py_sent.wer != 0:
                continue
            beam = self.search(py_sent, force_gold, py_sent.wer == 0)
            beam.beam[0].cost += py_sent.wer
            scores.append(beam.score)
            probs.append(prob)
            beams.append(beam)
        weights = mix_weights(probs, scores)
        return max(zip(weights, beams))[1]

    cdef dict _count_feats(self, Beam p_beam, Beam g_beam):
        # TODO: Improve this...
        cdef int v = get_violation(p_beam, g_beam)
        if v < 0:
            return {}
        cdef Sentence* psent = p_beam.sent
        cdef Sentence* gsent = g_beam.sent

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
        for w in range(gsent.n):
            gold_state.parse[w].tag = gsent.tokens[w].tag
        for w in range(psent.n):
            pred_state.parse[w].tag = psent.tokens[w].tag
        cdef dict counts = {}
        for clas in range(self.nr_moves):
            counts[clas] = {}
        cdef bint seen_diff = False
        cdef Token* gword
        cdef Token* pword
        for i in range(max((pt, gt))):
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
