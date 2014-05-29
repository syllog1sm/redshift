from _state cimport *
from transitions cimport Transition
from sentence cimport Token


from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair
from libcpp.vector cimport vector

ctypedef pair[double, size_t] ScoredMove
ctypedef pair[size_t, Transition] Candidate

cdef class Beam:
    cdef State** parents
    cdef State** beam

    cdef priority_queue[ScoredMove] queue
    #cdef Transition** moves
    cdef vector[Candidate] moves

    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t nr_class
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full
    cdef bint is_finished

    #cdef int enqueue(self, size_t i, bint force_gold) except -1
    cdef int enqueue(self, size_t i, vector[Transition] moves) except -1
    cdef int extend(self) except -1
    cdef int fill_parse(self, Token* parse) except -1
