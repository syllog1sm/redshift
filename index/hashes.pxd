from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, int64_t
from ext.murmurhash cimport *
from ext.sparsehash cimport *
from redshift.memory cimport Pool


cdef class Index:
    cdef size_t i
    cdef dict table
    cdef dict reverse
    
    cpdef size_t lookup(self, bytes entry)
    cpdef bytes get_str(self, size_t code)
    cpdef save(self, path)
    cpdef load(self, path)


cdef class ScoresCache:
    cdef uint64_t i
    cdef uint64_t pool_size
    cdef size_t scores_size
    cdef Pool _pool
    cdef double** _arrays
    cdef dense_hash_map[uint64_t, size_t] _cache
    cdef size_t n_hit
    cdef size_t n_miss

    cdef double* lookup(self, size_t size, void* kernel, bint* success)
    cdef int _resize(self, size_t new_size)
