from transitions cimport TransitionSystem

from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair

from ext.murmurhash cimport *
from ext.sparsehash cimport *
from _fast_state cimport *

cdef class FastBeam:
    cdef TransitionSystem trans
    
    cdef FastState** parents
    cdef FastState** beam
    cdef int** valid
    cdef int** costs
    cdef set seen_states
    
    cdef size_t max_class
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t bsize
    cdef bint is_full
    cdef bint is_finished

    cdef int extend_states(self, double** scores) except -1

