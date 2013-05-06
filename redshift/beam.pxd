from _state cimport *
from transitions cimport TransitionSystem

from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair


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
    cdef Violation violn
    cdef priority_queue[pair[double, size_t]]* next_moves
    
    cdef State* gold
    cdef State** parents
    cdef State** beam
    cdef double** scores
    cdef int** costs
    cdef bint** valid
    
    cdef object upd_strat
    cdef size_t max_class
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full

    cdef Kernel* next_state(self, size_t i)
    cdef int cost_next(self, size_t i, size_t* heads, size_t* labels) except -1
    cdef int extend_states(self) except -1
    cdef bint check_violation(self)
    cdef int fill_parse(self, size_t* hist, size_t* heads, size_t* labels) except -1

