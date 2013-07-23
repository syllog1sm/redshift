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
    cdef State** parents
    cdef State** beam
    cdef int** valid
    cdef int** costs
    
    cdef size_t max_class
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full
    cdef bint is_finished

    cdef Kernel* next_state(self, size_t i, size_t* tags)
    cdef int extend_states(self, double** scores) except -1
    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd, bint* edits) except -1

