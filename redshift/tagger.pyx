from thinc.features.extractor cimport Extractor
from thinc.ml.learner cimport LinearModel
from thinc.ml.learner cimport W as weight_t

import index.hashes
cimport index.hashes
from .util import Config

from redshift.sentence cimport Input, Sentence
from index.lexicon cimport Lexeme
from cymem.cymem cimport Pool
from preshed.maps cimport PreshMap

from ._tagger_features cimport fill_context
from ._tagger_features import *

from libc.stdint cimport uint64_t, int64_t
from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair

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
        tagger.guide.end_train_iter(n, feat_thresh)
        random.shuffle(indices)
    tagger.guide.end_training()
    with open(path.join(model_dir, 'tagger.gz'), 'w') as file_:
        tagger.guide.dump(file_)
    index.hashes.save_pos_idx(path.join(model_dir, 'pos'))
    return tagger


cdef class Tagger:
    def __cinit__(self, model_dir):
        self.cfg = Config.read(model_dir, 'tagger')
        self._pool = Pool()
        self.extractor = Extractor(basic + clusters + case + orth, [],
                                   bag_of_words=[])
        self._features = <uint64_t*>self._pool.alloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>self._pool.alloc(context_size(), sizeof(size_t))

        self.beam_width = self.cfg.beam_width

        if path.exists(path.join(model_dir, 'pos')):
            index.hashes.load_pos_idx(path.join(model_dir, 'pos'))
        nr_tag = index.hashes.get_nr_pos()
        self.guide = LinearModel(nr_tag, self.extractor.nr_feat)
        if path.exists(path.join(model_dir, 'tagger.gz')):
            with open(path.join(model_dir, 'tagger.gz'), 'r') as file_:
                self.guide.load(file_)
        self._beam_scores = <weight_t**>self._pool.alloc(self.beam_width, sizeof(weight_t*))
        for i in range(self.beam_width):
            self._beam_scores[i] = <weight_t*>self._pool.alloc(nr_tag, sizeof(weight_t))

    cpdef int tag(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef TaggerBeam beam = TaggerBeam(self.beam_width, sent.n, self.guide.nr_class)
        cdef size_t p_idx
        cdef TagState* s
        for i in range(sent.n - 1):
            # Extend beam
            for j in range(beam.bsize):
                # At this point, beam.clas is the _last_ prediction, not the
                # prediction for this instance
                self._predict(i, beam.parents[j], sent, self._beam_scores[j])
            beam.extend_states(self._beam_scores)
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
        cdef TaggerBeam beam = TaggerBeam(self.beam_width, sent.n, nr_class)
        cdef TagState* gold = extend_state(NULL, 0, NULL, 0, tmp_mem)
        cdef MaxViolnUpd updater = MaxViolnUpd(nr_class)
        for i in range(sent.n - 1):
            # Extend gold
            self._predict(i, gold, sent, scores)
            gold = extend_state(gold, sent.tokens[i].tag, scores, nr_class, tmp_mem)
            # Extend beam
            for j in range(beam.bsize):
                # At this point, beam.clas is the _last_ prediction, not the
                # prediction for this instance
                self._predict(i, beam.parents[j], sent, self._beam_scores[j])
            beam.extend_states(self._beam_scores)
            updater.compare(beam.states[0], gold, i)
            self.guide.n_corr += (gold.clas == beam.states[0].clas)
            self.guide.total += 1
        if updater.delta != -1:
            counts = updater.count_feats(self._features, self._context, sent,
                                         self.extractor)
            self.guide.update(counts)

    cdef int _predict(self, size_t i, TagState* s, Sentence* sent, weight_t* scores):
        fill_context(self._context, sent, s.clas, get_p(s), i)
        cdef size_t n = self.extractor.extract(self._features, self._context)
        self.guide.score(scores, self._features, n)


cdef class MaxViolnUpd:
    cdef TagState* pred
    cdef TagState* gold
    cdef Sentence* sent
    cdef weight_t delta
    cdef int length
    cdef size_t nr_class
    cdef size_t tmp
    def __cinit__(self, size_t nr_class):
        self.delta = -1
        self.length = -1
        self.nr_class = nr_class

    cdef int compare(self, TagState* pred, TagState* gold, size_t i):
        delta = pred.score - gold.score
        if delta > self.delta:
            self.delta = delta
            self.pred = pred
            self.gold = gold
            self.length = i 

    cdef dict count_feats(self, uint64_t* feats, size_t* context, Sentence* sent,
                          Extractor extractor):
        if self.length == -1:
            return {}
        cdef TagState* g = self.gold
        cdef TagState* p = self.pred
        cdef int i = self.length
        cdef dict counts = {}
        for clas in range(self.nr_class):
            counts[clas] = {} 
        cdef size_t gclas, gprev, gprevprev
        cdef size_t pclas, pprev, prevprev
        while g != NULL and p != NULL and i >= 0:
            gclas = g.clas
            gprev = get_p(g)
            gprevprev = get_pp(g)
            pclas = p.clas
            pprev = get_p(p)
            pprevprev = get_pp(p)
            if gclas == pclas and pprev == gprev and gprevprev == pprevprev:
                g = g.prev
                p = p.prev
                i -= 1
                continue
            fill_context(context, sent, gprev, gprevprev, i)
            extractor.extract(feats, context)
            extractor.count(counts[g.clas], feats, 1.0)
            fill_context(context, sent, pprev, pprevprev, i)
            extractor.extract(feats, context)
            extractor.count(counts[p.clas], feats, -1.0)
            g = g.prev
            p = p.prev
            i -= 1
        return counts


cdef class TaggerBeam:
    def __cinit__(self, size_t k, size_t length, nr_tag=None):
        self.nr_class = nr_tag
        self.k = k
        self.t = 0
        self.bsize = 1
        self.is_full = self.bsize >= self.k
        self._pool = Pool()
        self.beam = Beam(self.nr_class, k)
        self.states = <TagState**>self._pool.alloc(k, sizeof(TagState*))
        self.parents = <TagState**>self._pool.alloc(k, sizeof(TagState*))
        cdef size_t i
        for i in range(k):
            self.parents[i] = extend_state(NULL, 0, NULL, 0, self._pool)

    @cython.cdivision(True)
    cdef int extend_states(self, weight_t** ext_scores) except -1:
        self.beam.fill(ext_scores)
        self.bsize = 0
        cdef size_t i
        cdef size_t clas
        cdef TagState* prev
        while self.bsize < self.k:
            i, clas = self.beam.pop()
            prev = self.parents[i]
            self.states[self.bsize] = extend_state(prev, clas, ext_scores[i],
                                                   self.nr_class, self._pool)
            self.bsize += 1
        for i in range(self.bsize):
            self.parents[i] = self.states[i]
        self.is_full = self.bsize >= self.k
        self.t += 1


cdef TagState* extend_state(TagState* s, size_t clas, weight_t* scores,
                            size_t nr_class, Pool pool):
    cdef weight_t score
    ext = <TagState*>pool.alloc(1, sizeof(TagState))
    ext.prev = s
    ext.clas = clas
    if s == NULL:
        ext.score = 0
        ext.length = 0
    else:
        ext.score = s.score + scores[clas]
        ext.length = s.length + 1
    return ext


cdef inline size_t get_p(TagState* s) nogil:
    return s.prev.clas if s.prev != NULL else 0


cdef inline size_t get_pp(TagState* s) nogil:
    return get_p(s.prev) if s.prev != NULL else 0
