from libc.stdint cimport uint64_t
from ext.sparsehash cimport * 


cdef Word* lookup(bytes word)

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
