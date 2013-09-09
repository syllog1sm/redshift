# cython: profile=True
from _fast_state cimport *
from transitions cimport TransitionSystem

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair
cimport cython

from cython.operator cimport preincrement as inc
from cython.operator cimport dereference as deref


cdef class FastBeam:
    def __cinit__(self, TransitionSystem trans, 
                  size_t k, size_t length):
        self.trans = trans
        self.length = length
        self.k = k
        self.i = 0
        self.t = 0
        self.bsize = 1
        self.is_finished = False
        self.is_full = self.bsize >= self.k
        cdef size_t i
        self.parents = <FastState**>malloc(k * sizeof(FastState*))
        self.beam = <FastState**>malloc(k * sizeof(FastState*))
        self.seen_states = set()
        for i in range(k):
            self.beam[i] = init_fast_state()
            self.parents[i] = self.beam[i]
            self.seen_states.add(<size_t>self.parents[i])
        self.valid = <int**>malloc(self.k * sizeof(int*))
        self.costs = <int**>malloc(self.k * sizeof(int*))
        for i in range(self.k):
            self.valid[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))
            self.costs[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))

    @cython.cdivision(True)
    cdef int extend_states(self, double** ext_scores) except -1:
        # Former states are now parents, beam will hold the extensions
        cdef FastState** parents = self.parents
        cdef size_t parent_idx, clas, move_id
        cdef double* scores
        cdef priority_queue[pair[double, size_t]] next_moves
        next_moves = priority_queue[pair[double, size_t]]()
        # Get best parent/clas pairs by score
        cdef FastState* parent
        for parent_idx in range(self.bsize):
            parent = self.parents[parent_idx]
            scores = ext_scores[parent_idx]
            for clas in range(self.trans.nr_class):
                if self.valid[parent_idx][clas] != -1:
                    score = parent.score + scores[clas]
                    move_id = (parent_idx * self.trans.nr_class) + clas
                    next_moves.push(pair[double, size_t](score, move_id))
        cdef pair[double, size_t] data
        # Apply extensions for best continuations
        cdef uint64_t key
        cdef size_t i
        self.bsize = 0
        while self.bsize < self.k and not next_moves.empty():
            data = next_moves.top()
            i = data.second / self.trans.nr_class
            clas = data.second % self.trans.nr_class
            parent = self.parents[i]
            self.beam[self.bsize] = extend_fstate(parent, self.trans.moves[clas],
                                                  self.trans.labels[clas],
                                                  clas, ext_scores[i][clas],
                                                  self.costs[i][clas])
            self.seen_states.add(<size_t>self.beam[self.bsize])
            self.bsize += 1
            next_moves.pop()
        for i in range(self.bsize):
            self.parents[i] = self.beam[i]
        self.is_full = self.bsize >= self.k
        self.t += 1
        self.is_finished = is_finished(&self.beam[0].knl, self.length)
        assert self.t < (self.length * 3)

    def __dealloc__(self):
        cdef FastState* s
        cdef size_t addr
        for addr in self.seen_states:
            s = <FastState*>addr
            free(s)
        for i in range(self.k):
            free(self.valid[i])
            free(self.costs[i])
        free(self.beam)
        free(self.parents)
        free(self.valid)
        free(self.costs)
