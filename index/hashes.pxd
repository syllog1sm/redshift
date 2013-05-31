from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, int64_t

DEF VOCAB_SIZE = 1e6
DEF TAG_SET_SIZE = 100


cdef extern from "sparsehash/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        uint64_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin()
        iterator end()
        uint64_t size()
        uint64_t max_size()
        bint empty()
        uint64_t bucket_count()
        uint64_t bucket_size(uint64_t i)
        uint64_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(uint64_t n)
        void rehash(uint64_t n)
        dense_hash_map()
        dense_hash_map(uint64_t n)
        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        uint64_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32(void * key, uint64_t len, uint64_t seed, void* out)
    void MurmurHash3_x86_128(void * key, uint64_t len, uint64_t seed, void* out)

cdef extern from "MurmurHash2.h":
    uint64_t MurmurHash64A(void * key, uint64_t len, int64_t seed)
    uint64_t MurmurHash64B(void * key, uint64_t len, int64_t seed)


cdef class Index:
    cdef object path
    cdef bint save_entries
    cdef object out_file
    cdef uint64_t i

    cpdef set_path(self, path)
    cpdef save_entry(self, int i, object feat_str, object hashed, object value)
    cpdef save(self)
    cpdef load(self, path)


cdef class StrIndex(Index):
    cdef object vocab
    cdef dense_hash_map[uint64_t, uint64_t] table
    cdef uint64_t encode(self, char* feature) except 0
    cpdef load_entry(self, uint64_t i, object key, uint64_t hashed, uint64_t value)


cdef struct Cluster:
    size_t prefix
    size_t full


cdef class ClusterIndex:
    cdef Cluster* table
    cdef size_t prefix_len
    cdef size_t thresh
    cdef size_t n

cdef class PruningFeatIndex(Index):
    cdef uint64_t n
    cdef uint64_t p_i
    cdef uint64_t threshold
    cdef bint count_features
    cdef vector[dense_hash_map[uint64_t, uint64_t]] tables
    cdef vector[dense_hash_map[uint64_t, uint64_t]] unpruned
    cdef dense_hash_map[uint64_t, uint64_t] freqs
    cdef uint64_t encode(self, uint64_t* feature, uint64_t length, uint64_t i)
    cpdef load_entry(self, uint64_t i, object key, uint64_t hashed, uint64_t value)


#cdef class FeatIndex(Index):
#    cdef uint64_t n
#    cdef uint64_t threshold
#    cdef bint count_features
#    cdef vector[dense_hash_map[uint64_t, uint64_t]] tables
#    cdef uint64_t encode(self, uint64_t* feature, uint64_t length, uint64_t i)
#    cpdef load_entry(self, uint64_t i, object key, uint64_t hashed, uint64_t value)


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

#cdef uint64_t encode_feat(uint64_t* feature, uint64_t length, uint64_t i)

#cdef FeatIndex get_feat_idx()

cdef class InstanceCounter:
    cdef uint64_t n
    cdef vector[dense_hash_map[long, long]] counts_by_class
    cdef uint64_t add(self, uint64_t class_, uint64_t sent_id, uint64_t* history,
                  bint freeze_count) except 0
