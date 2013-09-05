from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, int64_t
from ext.murmurhash cimport *
from ext.sparsehash cimport *

DEF VOCAB_SIZE = 1e6
DEF TAG_SET_SIZE = 100

cdef class StrIndex:
    cdef size_t i
    cdef bint save_entries
    cdef object vocab
    cdef object case_stats
    cdef object strings
    cdef dense_hash_map[uint64_t, uint64_t] table
    cdef uint64_t encode(self, char* feature) except 0


cdef struct Cluster:
    size_t prefix4
    size_t prefix6
    size_t full


cdef class ClusterIndex:
    cdef Cluster* table
    cdef size_t prefix_len
    cdef size_t thresh
    cdef size_t n

cdef class ScoresCache:
    cdef uint64_t i
    cdef uint64_t pool_size
    cdef size_t scores_size
    cdef double** _pool
    cdef dense_hash_map[uint64_t, size_t] _cache
    cdef size_t n_hit
    cdef size_t n_miss

    cdef double* lookup(self, size_t size, void* kernel, bint* success)
    cdef int _resize(self, size_t new_size)

cpdef encode_word(object word)

cpdef int get_freq(object word) except -1

cdef StrIndex _pos_idx
cdef StrIndex _word_idx
cdef StrIndex _label_idx
cdef ClusterIndex _cluster_idx

cdef class InstanceCounter:
    cdef uint64_t n
    cdef vector[dense_hash_map[long, long]] counts_by_class
    cdef uint64_t add(self, uint64_t class_, uint64_t sent_id, uint64_t* history,
                  bint freeze_count) except 0
