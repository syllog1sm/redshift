# cython: profile=True
from _state cimport *

from transitions cimport Transition, transition

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair

cimport cython


cdef class Beam:
    def __cinit__(self, size_t k, size_t length, size_t moves_addr, size_t nr_class):
        self.length = length
        self.nr_class = nr_class
        self.k = k
        self.i = 0
        self.is_finished = False
        cdef size_t i
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        self.moves = <Transition**>malloc(k * sizeof(Transition*))
        cdef Transition* moves = <Transition*>moves_addr
        for i in range(k):
            self.parents[i] = init_state(length)
            self.beam[i] = init_state(length)
            self.moves[i] = <Transition*>calloc(self.nr_class, sizeof(Transition))
            for j in range(self.nr_class):
                assert moves[j].clas < nr_class
                self.moves[i][j].clas = moves[j].clas
                self.moves[i][j].move = moves[j].move
                self.moves[i][j].label = moves[j].label
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.queue = priority_queue[ScoredMove]()

    cdef int enqueue(self, size_t i, bint force_gold) except -1:
        cdef State* s = self.beam[i]
        cdef size_t move_id = i * self.nr_class
        if s.is_finished:
            self.queue.push(ScoredMove(s.score + (s.score / self.t), move_id))
            #self.queue.append((s.score + (s.score / self.t), i * self.nr_class))
            return 0
        cdef Transition* moves = self.moves[i]
        cdef Transition t
        cdef size_t j
        for j in range(self.nr_class):
            if moves[j].is_valid and (not force_gold or moves[j].cost == 0):
                self.queue.push(ScoredMove(s.score + moves[j].score, move_id + j))

    @cython.cdivision(True)
    cdef int extend(self):
        # Former states are now parents, beam will hold the extensions
        cdef State** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        self.psize = self.bsize
        self.bsize = 0
        cdef State* parent
        cdef State* s
        cdef Transition* t
        cdef ScoredMove data
        cdef size_t move_idx
        cdef size_t parent_idx
        while not self.queue.empty() and self.bsize < self.k:
            data = self.queue.top()
            parent_idx = data.second / self.nr_class
            move_idx = data.second % self.nr_class
            # We've got two arrays of states, and we swap beam-for-parents.
            # So, s here will get manipulated, then its beam will replace
            # parents later.
            copy_state(self.beam[self.bsize], self.parents[parent_idx])
            s = self.beam[self.bsize]
            s.score = data.first
            t = &self.moves[parent_idx][move_idx]
            if not s.is_finished:
                s.cost += t.cost
                transition(t, s)
                assert s.m != 0
            self.bsize += 1
            self.queue.pop()
        self.t += 1
        self.is_full = self.bsize >= self.k
        assert self.beam[0].m != 0
        for i in range(self.bsize):
            if not self.beam[i].is_finished:
                self.is_finished = False
                break
        else:
            self.is_finished = True
        while not self.queue.empty():
            self.queue.pop()

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
