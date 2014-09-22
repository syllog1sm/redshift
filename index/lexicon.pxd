from libc.stdint cimport uint64_t
from cymem.cymem cimport Pool
from trustyc.maps cimport PointerMap


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
    cdef Pool mem
    cdef PointerMap words
    cdef dict strings
    cdef size_t lookup(self, bytes word)
