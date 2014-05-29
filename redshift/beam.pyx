# cython: profile=True
from _state cimport *
from redshift.sentence cimport Input, Sentence, Token

from transitions cimport Transition, transition

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.queue cimport priority_queue
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref, preincrement as inc

cimport cython


cdef class Beam:
    def __cinit__(self, size_t k, Input py_sent):
        self.length = py_sent.length
        self.k = k
        self.i = 0
        self.t = 0
        self.bsize = 1
        self.psize = 0
        self.is_full = self.bsize >= self.k
        self.is_finished = False
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        cdef size_t i
        for i in range(k):
            self.parents[i] = init_state(py_sent.c_sent)
            self.beam[i] = init_state(py_sent.c_sent)
        self.queue = priority_queue[ScoredMove]()
        self.moves = vector[Candidate]()

    cdef int enqueue(self, size_t i, vector[Transition] moves) except -1:
        # TODO: Deal with is-final case
        cdef State* s = self.beam[i]
        cdef vector[Transition].iterator it = moves.begin()
        cdef Transition trans
        cdef size_t j = self.moves.size()
        while it != moves.end():
            trans = deref(it)
            self.moves.push_back(Candidate(i, trans))
            self.queue.push(ScoredMove(s.score + trans.score, j))
            j += 1
            inc(it)

    cdef int extend(self):
        cdef:
            ScoredMove scored_move
            State* parent
            State* s
            Transition t
        # Former states are now parents, beam will hold the extensions
        cdef State** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        self.psize = self.bsize
        self.bsize = 0
        
        self.is_finished = True
        while not self.queue.empty() and self.bsize < self.k:
            scored_ext = self.queue.top()
            candidate = self.moves[scored_ext.second]
            parent = self.parents[candidate.first]
            t = candidate.second
            # We've got two arrays of states, and we swap beam-for-parents.
            # So, s here will get manipulated, then its beam will replace
            # parents later.
            new_state = self.beam[self.bsize]
            copy_state(new_state, parent)
            new_state.score = scored_ext.first
            if not is_final(new_state):
                new_state.cost += t.cost
                transition(&t, new_state)
                assert new_state.m != 0
                self.is_finished = False
            self.bsize += 1
            self.queue.pop()
        self.t += 1
        self.is_full = self.bsize >= self.k
        
        while not self.queue.empty():
            self.queue.pop()
        self.moves.clear()

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
        for i in range(self.k):
            free_state(self.beam[i])
            free_state(self.parents[i])
        free(self.beam)
        free(self.parents)
