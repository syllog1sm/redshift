# cython: profile=True
from _state cimport *
from redshift.sentence cimport Input, Sentence, Token

from cymem.cymem cimport Pool

from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair

cimport cython


include "compile_time_options.pxi"
IF TRANSITION_SYSTEM == 'arc_eager':
    from .arc_eager cimport *
ELSE:
    from .arc_hybrid cimport *


cdef class Beam:
    def __cinit__(self, size_t k, size_t moves_addr, size_t nr_class, Input py_sent):
        self.length = py_sent.length
        self.nr_class = nr_class
        self.k = k
        self.i = 0
        self.is_finished = False
        self._pool = Pool()
        self.parents = <State**>self._pool.alloc(k, sizeof(State*))
        self.beam = <State**>self._pool.alloc(k, sizeof(State*))
        self.moves = <Transition**>self._pool.alloc(k, sizeof(Transition*))
        cdef Transition* moves = <Transition*>moves_addr
        cdef size_t i, j
        for i in range(k):
            self.parents[i] = init_state(py_sent.c_sent, self._pool)
            self.beam[i] = init_state(py_sent.c_sent, self._pool)
            self.moves[i] = <Transition*>self._pool.alloc(self.nr_class, sizeof(Transition))
            for j in range(self.nr_class):
                assert moves[j].clas < nr_class
                self.moves[i][j].clas = moves[j].clas
                self.moves[i][j].move = moves[j].move
                self.moves[i][j].label = moves[j].label
                self.moves[i][j].is_valid = True
                self.moves[i][j].score = 0
                self.moves[i][j].cost == 0
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.queue = priority_queue[ScoredMove]()

    cdef int enqueue(self, size_t i, bint force_gold) except -1:
        cdef State* s = self.beam[i]
        cdef size_t move_id = i * self.nr_class
        if is_final(s):
            self.queue.push(ScoredMove(s.score + (s.score / self.t), move_id))
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
            if not is_final(s):
                s.cost += t.cost
                transition(t, s)
            self.bsize += 1
            self.queue.pop()
        self.t += 1
        self.is_full = self.bsize >= self.k
        for i in range(self.bsize):
            if not is_final(self.beam[i]):
                self.is_finished = False
                break
        else:
            self.is_finished = True
        while not self.queue.empty():
            self.queue.pop()

    cdef int fill_parse(self, Token* parse) except -1:
        cdef size_t i, head 
        cdef State* s = self.beam[0]
        for i in range(1, s.n-1):
            head = i
            while s.parse[head].head != head and \
                  s.parse[head].head < (s.n-1) and \
                  s.parse[head].head != 0:
                head = s.parse[head].head
            s.parse[i].sent_id = head
        # No need to copy heads for root and start symbols
        for i in range(1, self.length - 1):
            parse[i] = s.parse[i]
 
    def __dealloc__(self):
        pass
