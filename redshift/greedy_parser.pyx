cimport cython
import random
import os.path
from os.path import join as pjoin
import shutil

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

from _state cimport *
from sentence cimport Sentence
from sentence cimport PySentence
from sentence import get_labels
from sentence cimport PySentence, Sentence
from transitions cimport TransitionSystem, transition_to_str 
from beam cimport Beam
from tagger cimport BeamTagger

from features.extractor cimport Extractor
import _parse_features
from _parse_features cimport *

import index.hashes
cimport index.hashes

from learn.perceptron cimport Perceptron

from libc.stdint cimport uint64_t, int64_t


VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50

from .parser cimport BaseParser

cdef double FOLLOW_ERR_PC = 0.90
cdef class GreedyParser(BaseParser):
    cdef int parse(self, Sentence* sent) except -1:
        cdef State* s
        cdef uint64_t* feats
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        if self.auto_pos:
            self.tagger.tag(sent)
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            feats = self._extract(sent, &s.kernel)
            self.moves.fill_valid(s, self.moves._costs)
            clas = self._predict(feats, self.moves._costs,
                                 &s.guess_labels[s.i])
            self.moves.transition(clas, s)
        # No need to copy heads for root and start symbols
        cdef size_t i
        for i in range(1, sent.length - 1):
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]
        for i in range(s.t):
            sent.parse.moves[i] = s.history[i]
        sent.parse.n_moves = s.t
        sent.parse.score = s.score
        fill_edits(s, sent.parse.edits)
        free_state(s)
 
    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef State* s = init_state(sent.length)
        cdef size_t pred
        cdef uint64_t* feats
        cdef size_t _ = 0

        cdef size_t* bu_tags 
        if self.auto_pos:
            bu_tags = <size_t*>calloc(sent.length, sizeof(size_t))
            memcpy(bu_tags, sent.pos, sent.length * sizeof(size_t))
            self.tagger.tag(sent)
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            self.moves.fill_valid(s, valid)
            feats = self._extract(sent, &s.kernel)
            pred = self._predict(feats, valid, &s.guess_labels[s.i])
            costs = self.moves.get_costs(s, sent.pos, sent.parse.heads,
                                         sent.parse.labels, sent.parse.edits,
                                         sent.parse.sbd)
            gold = pred if costs[pred] == 0 else self._predict(feats, costs, &_)
            self.guide.update(pred, gold, feats, 1)
            if iter_num >= 2 and random.random() < FOLLOW_ERR_PC:
                self.moves.transition(pred, s)
            else:
                self.moves.transition(gold, s)
            self.guide.n_corr += (gold == pred)
            self.guide.total += 1
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags)
        free_state(s)
        free(valid)

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef State* s = init_state(sent.length)
        cdef size_t pred
        cdef uint64_t* feats
        cdef size_t _ = 0
        cdef size_t* bu_tags 
        if self.auto_pos:
            bu_tags = <size_t*>calloc(sent.length, sizeof(size_t))
            memcpy(bu_tags, sent.pos, sent.length * sizeof(size_t))
            self.tagger.tag(sent)
 
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            feats = self._extract(sent, &s.kernel)
            self.moves.fill_valid(s, valid)
            pred = self._predict(feats, valid, &s.guess_labels[s.i])
            gold = self.moves.break_tie(s, sent.pos, sent.parse.heads,
                                         sent.parse.labels, sent.parse.edits,
                                         sent.parse.sbd)
            self.guide.update(pred, gold, feats, 1)
            self.moves.transition(gold, s)
            self.guide.n_corr += (gold == pred)
            self.guide.total += 1
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags) 
        free_state(s)
        free(valid)

    def say_config(self):
        if self.moves.allow_reattach and self.moves.allow_reduce:
            print 'NM L+D'
        elif self.moves.allow_reattach:
            print 'NM L'
        elif self.moves.allow_reduce:
            print 'NM D'

    cdef uint64_t* _extract(self, Sentence* sent, Kernel* kernel):
        fill_context(self._context, self.moves.n_labels, sent, kernel)
        self.extractor.extract(self._features, self._context)
        return self._features
   
    cdef int _predict(self, uint64_t* feats, int* valid, size_t* rlabel) except -1:
        cdef:
            size_t i
            double score
            size_t clas, best_valid, best_right
            double* scores

        cdef size_t right_move = 0
        cdef double valid_score = -10000
        cdef double right_score = -10000
        scores = self.guide.scores
        self.guide.fill_scores(self._features, scores)
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


cdef class GreedyTagger(BaseTagger):
    cdef int tag(self, Sentence* sent) except -1:
        cdef size_t i, clas, lookup
        cdef double incumbent, runner_up, score
        cdef size_t prev = sent.pos[0]
        cdef size_t alt = sent.pos[0]
        cdef size_t prevprev = 0
        for i in range(1, sent.length - 1):
            lookup = self.tagdict[sent.words[i]]
            if lookup != 0:
                sent.pos[i] = lookup
                sent.alt_pos[i] = 0
                alt = 0
                prevprev = prev
                prev = lookup
                continue 
            sent.pos[i] = 0
            sent.alt_pos[i] = 0
            fill_context(self._context, sent, prev, prevprev, alt, i)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, self.guide.scores)
            incumbent = -10000
            runner_up = -10000
            for clas in range(self.guide.nr_class):
                score = self.guide.scores[clas]
                if score >= incumbent:
                    sent.alt_pos[i] = sent.pos[i]
                    sent.pos[i] = clas
                    runner_up = incumbent
                    incumbent = score
            prevprev = prev
            prev = sent.pos[i]
            alt = sent.alt_pos[i]

    cdef int train_sent(self, Sentence* sent) except -1:
        cdef size_t w, clas, second, pred, prev, prevprev, lookup
        cdef double score, incumbent, runner_up
        cdef double second_score
        prev = sent.pos[0]
        alt = sent.pos[0]
        for w in range(1, sent.length - 1):
            lookup = self.tagdict[sent.words[w]]
            if lookup != 0:
                alt = 0
                prevprev = prev
                prev = lookup
                continue 
            fill_context(self._context, sent, prev, prevprev, alt, w)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, self.guide.scores)
            incumbent = 0
            runner_up = 0
            pred = 0
            second = 0
            for clas in range(self.nr_tag):
                score = self.guide.scores[clas]
                if score >= incumbent:
                    runner_up = incumbent
                    second = pred
                    incumbent = score
                    pred = clas
            if pred != sent.pos[w]:
                self.guide.update(pred, sent.pos[w], self._features, 1.0)
            else:
                self.guide.n_corr += 1
            self.guide.total += 1
            prevprev = prev
            prev = pred
            alt = second



