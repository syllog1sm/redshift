from _state cimport *
from transitions cimport Transition
from sentence cimport Token
from sentence cimport Step


from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair
from libcpp.vector cimport vector

ctypedef pair[double, size_t] ScoredMove
ctypedef pair[size_t, Transition] Candidate
ctypedef Transition* History


cdef class Beam:
    cdef Step* lattice
    cdef Sentence* sent

    cdef State** parents
    cdef State** beam

    cdef Transition** moves
    cdef vector[History] history
    cdef vector[double] scores
    cdef vector[int] costs
    cdef vector[size_t] lengths

    cdef double beta
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t nr_class
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full
    cdef bint is_finished

    cdef int extend(self) except -1
    cdef int fill_parse(self, Token* parse) except -1


cdef int get_violation(Beam pred, Beam gold)
