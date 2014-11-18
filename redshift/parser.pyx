# cython: profile=True
"""
MALT-style dependency parser
"""
cimport cython
import random
import os.path
from os.path import join as pjoin
import shutil
import json

from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool, Address
from thinc.typedefs cimport weight_t, class_t, feat_t, atom_t
from thinc.search cimport Beam, MaxViolation

from _state cimport *
from sentence cimport Input, Sentence, Token, Step

from tagger cimport Tagger
from util import Config

from thinc.features cimport Extractor
from thinc.features cimport ConjFeat
import _parse_features
from _parse_features cimport *

import index.hashes
cimport index.hashes

from thinc.learner cimport LinearModel


include "compile_time_options.pxi"
IF TRANSITION_SYSTEM == 'arc_eager':
    from .arc_eager cimport *
ELSE:
    from .arc_hybrid cimport *


VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50


DEBUG = False 
def set_debug(val):
    global DEBUG
    DEBUG = val


def train(train_str, model_dir, n_iter=15, beam_width=8, train_tagger=True,
          feat_set='basic', feat_thresh=10, seed=0,
          use_edit=False, use_break=False, use_filler=False):
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    cdef list sents = [Input.from_conll(s) for s in
                       train_str.strip().split('\n\n') if s.strip()]
    left_labels, right_labels, dfl_labels = get_labels(sents)
    Config.write(model_dir, 'config', beam_width=beam_width, features=feat_set,
                 feat_thresh=feat_thresh, seed=seed,
                 left_labels=left_labels, right_labels=right_labels,
                 dfl_labels=dfl_labels, use_break=use_break)
    Config.write(model_dir, 'tagger', beam_width=4, features='basic',
                 feat_thresh=5, tags={})
    parser = Parser(model_dir)
    indices = list(range(len(sents)))
    cdef Input py_sent
    for n in range(n_iter):
        for i in indices:
            py_sent = sents[i]
            parser.tagger.train_sent(py_sent)
            parser.train_sent(py_sent)
        acc = float(parser.guide.n_corr) / parser.guide.total
        print(parser.guide.end_train_iter(n, feat_thresh) + '\t' +
              parser.tagger.guide.end_train_iter(n, feat_thresh))
        random.shuffle(indices)
    parser.guide.end_training()
    parser.tagger.guide.end_training()
    parser.guide.dump(pjoin(model_dir, 'model'))
    parser.tagger.guide.dump(pjoin(model_dir, 'tagger'))
    index.hashes.save_pos_idx(pjoin(model_dir, 'pos'))
    index.hashes.save_label_idx(pjoin(model_dir, 'labels'))
    return acc


def get_labels(sents):
    left_labels = set()
    right_labels = set()
    dfl_labels = set()
    cdef Input sent
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            if sent.c_sent.tokens[j].is_edit:
                dfl_labels.add(sent.c_sent.tokens[j].label)
            elif sent.c_sent.tokens[j].head > j:
                left_labels.add(sent.c_sent.tokens[j].label)
            else:
                right_labels.add(sent.c_sent.tokens[j].label)
    return list(sorted(left_labels)), list(sorted(right_labels)), list(sorted(dfl_labels))


def get_templates(feats_str):
    match_feats = []
    # This value comes out of compile_time_options.pxi
    IF TRANSITION_SYSTEM == 'arc_eager':
        templates = _parse_features.arc_eager
    ELSE:
        templates = _parse_features.arc_hybrid
    if 'disfl' in feats_str:
        templates += _parse_features.disfl
        templates += _parse_features.new_disfl
        templates += _parse_features.suffix_disfl
        templates += _parse_features.extra_labels
        templates += _parse_features.clusters
        templates += _parse_features.edges
        templates += _parse_features.prev_next
        match_feats = _parse_features.match_templates()
    elif 'clusters' in feats_str:
        templates += _parse_features.clusters
    if 'bitags' in feats_str:
        templates += _parse_features.pos_bigrams()
    return templates, [ConjFeat for _ in templates]


cdef class Parser:
    cdef object cfg
    cdef Pool _pool
    cdef Extractor extractor
    cdef LinearModel guide
    cdef Tagger tagger
    cdef Transition* moves
    cdef feat_t* _features
    cdef weight_t* _values
    cdef atom_t* _context
    cdef size_t nr_moves

    def __init__(self, model_dir):
        assert os.path.exists(model_dir) and os.path.isdir(model_dir)
        self.cfg = Config.read(model_dir, 'config')
        self.extractor = Extractor(*get_templates(self.cfg.features))
        self._pool = Pool()
        self._features = <feat_t*>self._pool.alloc(self.extractor.n, sizeof(feat_t))
        self._values = <weight_t*>self._pool.alloc(self.extractor.n, sizeof(weight_t))
        self._context = <atom_t*>self._pool.alloc(_parse_features.context_size(), sizeof(atom_t))

        if os.path.exists(pjoin(model_dir, 'labels')):
            index.hashes.load_label_idx(pjoin(model_dir, 'labels'))
        self.nr_moves = get_nr_moves(self.cfg.left_labels, self.cfg.right_labels,
                                     self.cfg.dfl_labels, self.cfg.use_break)
        self.moves = <Transition*>self._pool.alloc(self.nr_moves, sizeof(Transition))
        fill_moves(self.cfg.left_labels, self.cfg.right_labels, self.cfg.dfl_labels,
                   self.cfg.use_break, self.moves)
        
        self.guide = LinearModel(self.nr_moves)
        if os.path.exists(pjoin(model_dir, 'model')):
            self.guide.load(pjoin(model_dir, 'model'))
        if os.path.exists(pjoin(model_dir, 'pos')):
            index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        self.tagger = Tagger(model_dir)

    cpdef int parse(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef Token* gold_parse = sent.tokens
        if self.tagger:
            self.tagger.tag(py_sent)
        cdef Beam beam = Beam(self.nr_moves, self.cfg.beam_width)
        beam.initialize(_init_callback, sent.n, sent)
        self.guide.cache.flush()
        cdef int i
        while not beam.is_done:
            self._advance_beam(beam, NULL, False)
        _fill_parse(sent.tokens, <State*>beam.at(0))
        sent.score = beam.score

    cdef int train_sent(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef Address tags_mem = Address(sent.n, sizeof(size_t))
        cdef size_t* gold_tags = <size_t*>tags_mem.ptr
        cdef Token* gold_parse = sent.tokens
        cdef int i
        #print py_sent.words
        for i in range(sent.n):
            gold_tags[i] = gold_parse[i].tag
        if self.tagger:
            self.tagger.tag(py_sent)
        cdef Beam p_beam = Beam(self.nr_moves, self.cfg.beam_width)
        cdef Beam g_beam = Beam(self.nr_moves, self.cfg.beam_width)
        p_beam.initialize(_init_callback, sent.n, sent)
        g_beam.initialize(_init_callback, sent.n, sent)

        cdef MaxViolation violn = MaxViolation()

        self.guide.cache.flush()
        cdef Transition* m
        cdef State* state
        while not p_beam.is_done and not g_beam.is_done:
            self._advance_beam(p_beam, gold_parse, False)
            self._advance_beam(g_beam, gold_parse, True)
            violn.check(p_beam, g_beam)
       
        counts = {}
        if violn.delta >= 0:
            self._count_feats(counts, sent, violn.g_hist, 1)
            self._count_feats(counts, sent, violn.p_hist, -1)
            self.guide.update(counts)
        else:
            self.guide.update({})
        t = 0
        for clas, clas_counts in counts.items():
            for c, f in clas_counts.items():
                t += abs(f)
        for i in range(sent.n):
            sent.tokens[i].tag = gold_tags[i]
        self.guide.n_corr += violn.cost == 0
        self.guide.total += 1

    cdef int _advance_beam(self, Beam beam, Token* gold_parse, bint follow_gold) except -1:
        cdef int i, j
        for i in range(beam.size):
            state = <State*>beam.at(i)
            if is_final(state):
                continue
            if gold_parse != NULL:
                fill_costs(state, self.moves, self.nr_moves, gold_parse)
            if not follow_gold:
                fill_valid(state, self.moves, self.nr_moves)
            self._predict(state, self.moves)
            for j in range(self.nr_moves):
                m = &self.moves[j]
                beam.set_cell(i, j, m.score, m.is_valid, m.cost)
        beam.advance(_transition_callback, self.moves)
        beam.check_done(_is_done_callback, NULL)

    cdef int _predict(self, State* s, Transition* classes) except -1:
        if is_final(s):
            return 0
        cdef bint cache_hit = False
        fill_slots(s)
        scores = self.guide.cache.lookup(sizeof(SlotTokens), &s.slots, &cache_hit)
        if not cache_hit:
            fill_context(self._context, &s.slots, s.parse)
            nr_active = self.extractor.extract(self._features, self._values,
                                               self._context, NULL)
            self.guide.score(scores, self._features, self._values)
        cdef size_t i
        for i in range(self.nr_moves):
            classes[i].score = scores[i]

    cdef dict _count_feats(self, dict counts, Sentence* sent, list hist, int inc):
        cdef Pool mem = Pool()
        cdef State* state = init_state(sent, mem)
        cdef class_t clas
        for clas in hist:
            fill_slots(state)
            fill_context(self._context, &state.slots, state.parse)
            self.extractor.extract(self._features, self._values, self._context, NULL)
            self.extractor.count(counts.setdefault(clas, {}), self._features, inc)
            transition(&self.moves[clas], state)
        for clas, class_counts in list(counts.items()):
            pruned = {}
            for feat, feat_count in class_counts.items():
                if abs(feat_count) >= 1:
                    pruned[feat] = feat_count
            counts[clas] = pruned


cdef int _fill_parse(Token* parse, State* s) except -1:
    cdef int i, head 
    for i in range(1, s.n-1):
        head = i
        while s.parse[head].head != head and \
                s.parse[head].head < (s.n-1) and \
                s.parse[head].head != 0:
            head = s.parse[head].head
        s.parse[i].sent_id = head
    # No need to copy heads for root and start symbols
    for i in range(1, s.n - 1):
        parse[i] = s.parse[i]


cdef void* _init_callback(Pool mem, int n, void* extra_args):
    return init_state(<Sentence*>extra_args, mem)


cdef int _transition_callback(void* dest, void* src, class_t clas, void* extra_args):
    state = <State*>dest
    parent = <State*>src
    moves = <Transition*>extra_args
    copy_state(state, parent)
    transition(&moves[clas], state)


cdef int _is_done_callback(void* state, void* extra_args):
    return is_final(<State*>state)
