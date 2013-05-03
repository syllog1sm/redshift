from _state cimport *
from transitions cimport TransitionSystem

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue


cdef class Violation:
    """
    A gold/prediction pair where the g.score < p.score
    """
    cdef size_t t
    cdef size_t* ghist
    cdef size_t* phist
    cdef double delta
    cdef int cost
    cdef bint out_of_beam
    
    cdef int set(self, State*p, State* g, bint out_of_beam) except -1

cdef class Beam:
    cdef TransitionSystem trans
    cdef State** parents
    cdef State** beam
    cdef State* gold
    cdef priority_queue[pair[double, size_t]]* next_moves
    cdef Cont** conts
    cdef object upd_strat
    cdef size_t max_class
    cdef size_t k
    cdef size_t i
    cdef size_t bsize
    cdef size_t psize
    cdef Violation violn
    cdef bint is_full

    cdef int add(self, size_t par_idx, double score, int cost,
                 size_t clas, size_t rlabel) except -1
    cdef int extend(self, size_t parent_idx, double* scores) except -1
    cdef bint check_violation(self)
    cdef State* best_p(self) except NULL
    cdef refresh(self)

