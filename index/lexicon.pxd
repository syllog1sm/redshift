from libc.stdint cimport uint64_t
from ext.sparsehash cimport * 


cpdef size_t lookup(bytes word)

cpdef bytes get_str(size_t word)

cdef struct Lexeme:
    size_t orig
    size_t norm
    size_t cluster
    size_t prefix
    size_t suffix
    bint oft_upper
    bint oft_title
    bint non_alpha


cdef Lexeme BLANK_WORD

cdef class Lexicon:
    cdef dense_hash_map[uint64_t, size_t] words
    cdef dict strings
    cdef size_t lookup(self, bytes word)
