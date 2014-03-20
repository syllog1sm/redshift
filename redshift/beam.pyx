# cython: profile=True
from _state cimport *

from transitions cimport Transition, transition

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t


cdef class Beam:
    def __cinit__(self, size_t k, size_t length, size_t nr_class):
        self.length = length
        self.nr_class = nr_class
        self.k = k
        self.i = 0
        self.is_finished = False
        cdef size_t i
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        self.moves = <Transition**>malloc(k * sizeof(Transition*))
        for i in range(k):
            self.parents[i] = init_state(length)
            self.beam[i] = init_state(length)
            self.moves[i] = <Transition*>calloc(self.nr_class, sizeof(Transition))
            for j in range(self.nr_class):
                self.moves[i][j] = self.trans.moves[j]
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.queue = []

    cdef int enqueue(self, size_t i, bint force_gold) except -1:
        cdef State* s = self.beam[i]
        if s.is_finished:
            self.queue.append((s.score + (s.score / self.t), i, -1))
            return 0
        cdef Transition* t
        for j in range(self.trans.nr_class):
            t = &self.moves[i][j]
            if t.is_valid and (t.cost == 0 or not force_gold):
                self.queue.append((s.score + t.score, i, j))

    cdef int extend(self):
        self.queue.sort()
        self.queue.reverse()
        # Former states are now parents, beam will hold the extensions
        cdef State** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        self.psize = self.bsize
        self.bsize = 0
        cdef State* parent
        cdef State* s
        cdef Transition* t
        for score, parent_idx, move_idx in self.queue[:self.k]:
            t = &self.moves[parent_idx][move_idx]
            parent = self.parents[parent_idx]
            # We've got two arrays of states, and we swap beam-for-parents.
            # So, s here will get manipulated, then copied into parents later.
            s = self.beam[self.bsize]
            copy_state(s, parent)
            s.score = score
            if not s.is_finished:
                s.cost += t.cost
                transition(t, s)
                self.bsize += 1
        self.t += 1
        self.is_full = self.bsize >= self.k
        self.queue = []
        for i in range(self.bsize):
            if not self.beam[i].is_finished:
                self.is_finished = False
                break
        else:
            self.is_finished = True

    cdef int fill_parse(self, AnswerToken* parse) except -1:
        cdef size_t i
        # No need to copy heads for root and start symbols
        for i in range(1, self.length - 1):
            parse[i] = self.beam[0].parse[i]
        #fill_edits(self.beam[0], edits)
 
    def __dealloc__(self):
        for i in range(self.k):
            free_state(self.beam[i])
            free_state(self.parents[i])
            free(self.moves[i])
        free(self.beam)
        free(self.parents)
