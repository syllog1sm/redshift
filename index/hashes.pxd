from libc.stdint cimport uint64_t, int64_t


cdef class Index:
    cdef size_t i
    cdef dict table
    cdef dict reverse
    
    cpdef size_t lookup(self, bytes entry)
    cpdef bytes get_str(self, size_t code)
    cpdef save(self, path)
    cpdef load(self, path)
