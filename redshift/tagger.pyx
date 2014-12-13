# cython: profile=True

import index.hashes
cimport index.hashes
from .util import Config

from redshift.sentence cimport Input, Sentence
from index.lexicon cimport Lexeme
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from ._tagger_features cimport fill_context
from ._tagger_features import *
from thinc.features cimport Feature, count_feats

cimport cython
import os
from os import path
import random
import shutil


def train(train_str, model_dir, beam_width=4, features='basic', nr_iter=10,
          feat_thresh=10):
    cdef Input sent
    cdef size_t i
    if path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    sents = [Input.from_pos(s) for s in train_str.strip().split('\n') if s.strip()]
    # Dict instead of set so json serialisable
    tags = {}
    for sent in sents:
        for i in range(sent.c_sent.n):
            tags[sent.c_sent.tokens[i].tag] = 1
    Config.write(model_dir, 'tagger', beam_width=beam_width, features=features,
                 feat_thresh=feat_thresh, tags=tags)
    tagger = Tagger(model_dir)
    indices = list(range(len(sents)))
    for n in range(nr_iter):
        for i in indices:
            sent = sents[i]
            tagger.train_sent(sent)
        print tagger.guide.end_train_iter(n, feat_thresh)
        random.shuffle(indices)
    tagger.guide.end_training()
    tagger.guide.dump(path.join(model_dir, 'tagger'))
    index.hashes.save_pos_idx(path.join(model_dir, 'pos'))
    return tagger


cdef class Tagger:
    def __init__(self, model_dir):
        self.cfg = Config.read(model_dir, 'tagger')
        self._pool = Pool()
        templates = basic + clusters + case + orth
        self.extractor = Extractor(templates)
        self._context = <atom_t*>self._pool.alloc(context_size(), sizeof(atom_t))

        self.beam_width = self.cfg.beam_width

        if path.exists(path.join(model_dir, 'pos')):
            index.hashes.load_pos_idx(path.join(model_dir, 'pos'))
        nr_tag = index.hashes.get_nr_pos()
        self.guide = LinearModel(nr_tag, self.extractor.n_templ)
        if path.exists(path.join(model_dir, 'tagger')):
            self.guide.load(path.join(model_dir, 'tagger'))
        self._beam_scores = <weight_t**>self._pool.alloc(self.beam_width, sizeof(weight_t*))
        for i in range(self.beam_width):
            self._beam_scores[i] = <weight_t*>self._pool.alloc(nr_tag, sizeof(weight_t))

    cpdef int tag(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef Beam beam = Beam(self.guide.nr_class, self.beam_width, sizeof(TagState))
        cdef size_t p_idx
        cdef TagState* s
        cdef size_t i, j
        for i in range(sent.n - 1):
            # Extend beam
            for j in range(beam.size):
                # At this point, beam.clas is the _last_ prediction, not the
                # prediction for this instance
                self._predict(i, <TagState*>beam.parents[j], sent, self._beam_scores[j])
            beam_extend(beam, self._beam_scores, 0)
        s = <TagState*>beam.states[0]
        cdef int t = sent.n - 1
        while t >= 1 and s.prev != NULL:
            t -= 1
            sent.tokens[t].tag = s.clas
            s = s.prev

    cdef int train_sent(self, Input py_sent) except -1:
        cdef size_t  i, j 
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t nr_class = self.guide.nr_class
        cdef weight_t* scores = self.guide.scores
        cdef Pool tmp_mem = Pool()
        cdef Beam beam = Beam(self.guide.nr_class, self.beam_width, sizeof(TagState))
        cdef TagState* gold = <TagState*>tmp_mem.alloc(1, sizeof(TagState))
        cdef MaxViolation violn = MaxViolation()
        cdef TagState* s
        for i in range(sent.n-1):
            # Extend gold
            self._predict(i, gold, sent, scores)
            gold = extend_state(gold, sent.tokens[i].tag, scores[sent.tokens[i].tag],
                                0, tmp_mem)
            # Extend beam
            for j in range(beam.size):
                # At this point, beam.clas is the _last_ prediction, not the
                # prediction for this instance
                self._predict(i, <TagState*>beam.parents[j], sent, self._beam_scores[j])
            beam_extend(beam, self._beam_scores, sent.tokens[i].tag)
            s = <TagState*>beam.states[0]
            violn.check(s.cost, s.score, gold.score, s, gold, i)
            self.guide.n_corr += (gold.clas == s.clas)
            self.guide.total += 1
        if violn.delta != -1:
            counts = self._count_feats(sent, <TagState*>violn.pred,
                                       <TagState*>violn.gold, violn.n)
            self.guide.update(counts)

    cdef int _predict(self, size_t i, TagState* s, Sentence* sent, weight_t* scores) except -1:
        fill_context(self._context, sent, s.clas, get_p(s), i)
        cdef int nr_active
        cdef Feature* feats
        feats = self.extractor.get_feats(self._context, &nr_active)
        self.guide.set_scores(scores, feats, nr_active)

    cdef dict _count_feats(self, Sentence* sent, TagState* p, TagState* g, int i):
        if i == -1:
            return {}
        cdef int nr_active
        cdef Feature* feats
        cdef atom_t* context = self._context
        cdef dict counts = {}
        for clas in range(self.guide.nr_class):
            counts[clas] = {} 
        cdef size_t gclas, gprev, gprevprev
        cdef size_t pclas, pprev, prevprev
        while g != NULL and p != NULL and i >= 0:
            gclas = g.clas
            assert g.clas >= 0
            gprev = get_p(g)
            gprevprev = get_pp(g)
            pclas = p.clas
            assert p.clas >= 0
            pprev = get_p(p)
            pprevprev = get_pp(p)
            if gclas == pclas and pprev == gprev and gprevprev == pprevprev:
                g = g.prev
                p = p.prev
                i -= 1
                continue
            fill_context(context, sent, gprev, gprevprev, i)
            feats = self.extractor.get_feats(context, &nr_active)
            count_feats(counts[g.clas], feats, nr_active, 1)

            fill_context(context, sent, pprev, pprevprev, i)
            feats = self.extractor.get_feats(context, &nr_active)
            count_feats(counts[pclas], feats, nr_active, -1)

            g = g.prev
            p = p.prev
            i -= 1
        return counts


cdef int beam_extend(Beam beam, weight_t** ext_scores, size_t gold) except -1:
    beam.fill(ext_scores)
    cdef int i
    cdef size_t clas
    cdef TagState* prev
    beam.size = 0
    while beam.size < beam.width:
        i, clas = beam.pop()
        prev = <TagState*>beam.parents[i]
        beam.states[beam.size] = extend_state(prev, clas, ext_scores[i][clas],
                                              clas != gold, beam.mem)
        beam.size += 1
    for i in range(beam.size):
        beam.parents[i] = beam.states[i]


cdef TagState* extend_state(TagState* s, size_t clas, weight_t score, size_t cost,
                             Pool pool):
    ext = <TagState*>pool.alloc(1, sizeof(TagState))
    ext.prev = s
    ext.clas = clas
    ext.score = s.score + score
    ext.length = s.length + 1
    ext.cost = s.cost + cost
    return ext


cdef inline size_t get_p(TagState* s) nogil:
    return s.prev.clas if s.prev != NULL else 0


cdef inline size_t get_pp(TagState* s) nogil:
    return get_p(s.prev) if s.prev != NULL else 0
