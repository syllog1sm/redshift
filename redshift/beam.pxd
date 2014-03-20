from _state cimport *
from transitions cimport Transition


cdef class Beam:
    cdef State** parents
    cdef State** beam

    cdef Transition** moves

    cdef list queue
    
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t nr_class
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full
    cdef bint is_finished

    cdef int enqueue(self, size_t i, bint force_gold) except -1
    cdef int extend(self) except -1
    cdef int fill_parse(self, AnswerToken* parse) except -1
