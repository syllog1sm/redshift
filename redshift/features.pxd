from libc.stdint cimport uint64_t, int64_t
from _state cimport State
#from index.hashes cimport FeatIndex
from io_parse cimport Sentence
from libcpp.vector cimport vector
from libcpp.utility cimport pair


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




cdef struct Predicate:
    int id, n, expected_size
    uint64_t* raws
    int* args

cdef class FeatureSet:
    cdef Predicate* predicates
    cdef size_t* context
    cdef uint64_t* features 
    cdef int n
    cdef int nr_label
    cdef uint64_t* extract(self, Sentence* sent, State* state)
    cdef bint save_entries
    cdef object out_file
    cdef uint64_t i
    cdef bint count_features
    cdef uint64_t threshold
    cdef size_t nr_uni
    cdef size_t nr_multi
    cdef size_t* uni_feats

    cdef dense_hash_map[uint64_t, uint64_t] unigrams
    cdef vector[dense_hash_map[uint64_t, uint64_t]] tables



cdef int CONTEXT_SIZE


cdef void fill_context(size_t* context, size_t nr_label, size_t n0, size_t n1, size_t n2,
                      size_t s0, size_t s1, size_t stack_len,
                      size_t* words, size_t* pos, size_t* browns,
                      size_t* heads, size_t* labels, size_t* l_vals, size_t* r_vals,
                      size_t* s0_lkids, size_t* s0_rkids, size_t* s1_lkids, size_t* s1_rkids,
                      size_t* n0_lkids,
                      bint* s0_llabels, bint* s0_rlabels, bint* n0_llabels)
