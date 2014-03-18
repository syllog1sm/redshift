from libc.stdint cimport uint64_t
from ext.sparsehash cimport * 


cpdef size_t lookup(bytes word)

cpdef bytes get_str(size_t word)

cpdef int add(bytes word) except -1

cdef struct Word:
    size_t orig
    size_t norm
    size_t cluster
    size_t prefix
    size_t suffix
    bint oft_upper
    bint oft_title
    bint non_alpha


cdef class Vocab:
    cdef dense_hash_map[uint64_t, size_t] words
    cdef dict strings
