# cython: profile=True
from _state cimport *
from transitions cimport TransitionSystem

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair
cimport cython

from cython.operator cimport preincrement as inc
from cython.operator cimport dereference as deref


cdef class Beam:
    def __cinit__(self, TransitionSystem trans, 
                  size_t k, size_t length):
        self.trans = trans
        self.length = length
        self.k = k
        self.i = 0
        self.is_finished = False
        cdef size_t i
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        for i in range(k):
            self.parents[i] = init_state(length)
        for i in range(k):
            self.beam[i] = init_state(length)
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.valid = <int**>malloc(self.k * sizeof(int*))
        self.costs = <int**>malloc(self.k * sizeof(int*))
        for i in range(self.k):
            self.valid[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))
            self.costs[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))

    cdef Kernel* next_state(self, size_t idx, size_t* tags):
        self.trans.fill_valid(self.beam[idx], self.valid[idx])
        fill_kernel(self.beam[idx], tags)
        return &self.beam[idx].kernel

    @cython.cdivision(True)
    cdef int extend_states(self, double** ext_scores) except -1:
        # Former states are now parents, beam will hold the extensions
        cdef State** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        self.psize = self.bsize
        self.bsize = 0
        cdef size_t parent_idx, clas, move_id
        cdef double mean_score, score
        cdef double* scores
        cdef priority_queue[pair[double, size_t]] next_moves = priority_queue[pair[double, size_t]]()
        # Get best parent/clas pairs by score
        cdef State* parent
        for parent_idx in range(self.psize):
            parent = self.parents[parent_idx]
            # Account for variable-length transition histories
            if parent.is_finished:
                move_id = (parent_idx * self.trans.nr_class) + 0
                mean_score = parent.score / self.t
                next_moves.push(pair[double, size_t](parent.score + mean_score, move_id))
                continue
            scores = ext_scores[parent_idx]
            r_score = scores[self.trans.r_start]
            parent.guess_labels[parent.i] = self.trans.labels[self.trans.r_start]
            for clas in range(self.trans.nr_class):
                if self.valid[parent_idx][clas] != -1:
                    score = parent.score + scores[clas]
                    move_id = (parent_idx * self.trans.nr_class) + clas
                    next_moves.push(pair[double, size_t](score, move_id))
                if scores[clas] >= r_score and self.trans.r_start < clas < self.trans.r_end:
                    r_score = scores[clas]
                    parent.guess_labels[parent.i] = self.trans.labels[clas]
        cdef pair[double, size_t] data
        # Apply extensions for best continuations
        cdef State* s
        cdef uint64_t key
        while self.bsize < self.k and not next_moves.empty():
            data = next_moves.top()
            parent_idx = data.second / self.trans.nr_class
            clas = data.second % self.trans.nr_class
            parent = self.parents[parent_idx]
            # We've got two arrays of states, and we swap beam-for-parents.
            # So, s here will get manipulated, then copied into parents later.
            s = self.beam[self.bsize]
            copy_state(s, parent)
            s.score = data.first
            if not s.is_finished:
                s.cost += self.costs[parent_idx][clas]
                self.trans.transition(clas, s)
            self.bsize += 1
            next_moves.pop()
        self.is_full = self.bsize >= self.k
        # Flush next_moves queue
        self.t += 1
        for i in range(self.bsize):
            if not self.beam[i].is_finished:
                self.is_finished = False
                break
        else:
            self.is_finished = True

    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd, bint* edits) except -1:
        cdef size_t i
        # No need to copy heads for root and start symbols
        for i in range(1, self.length - 1):
            assert self.beam[0].heads[i] != 0
            #tags[i] = self.beam[0].tags[i]
            heads[i] = self.beam[0].heads[i]
            labels[i] = self.beam[0].labels[i]
            # TODO: Do sentence boundary detection here
        fill_edits(self.beam[0], edits)
 
    def __dealloc__(self):
        for i in range(self.k):
            free_state(self.beam[i])
            free_state(self.parents[i])
            free(self.valid[i])
            free(self.costs[i])
        free(self.beam)
        free(self.parents)
        free(self.valid)
        free(self.costs)


