cimport index.hashes
from libcpp.vector cimport vector
from libcpp.utility cimport pair

from libc.stdint cimport int64_t, uint64_t

DEF MAX_PARAMS = 5000000

DEF MAX_DENSE = 100000

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


cdef struct Param:
    double w
    double acc
    size_t clas
    size_t last_upd


cdef struct DenseParams:
    double* w
    double* acc
    size_t* last_upd


cdef struct SquareFeature:
    DenseParams* parts    
    bint* seen
    size_t nr_seen


cdef struct DenseFeature:
    double* w
    double* acc
    size_t* last_upd
    uint64_t id
    size_t nr_seen
    size_t s
    size_t e

cdef class Perceptron:
    cdef int nr_class
    cdef double *scores
    cdef object path
    cdef bint is_trained
    cdef float n_corr
    cdef float total

    cdef size_t nr_raws
    cdef DenseFeature** raws

    cdef size_t div
    cdef uint64_t now
 
    cdef dense_hash_map[uint64_t, size_t] W
    cdef index.hashes.ScoresCache cache
    cdef bint use_cache

    cdef int add_feature(self, uint64_t f)
    cdef int add_instance(self, size_t label, double weight, int n, uint64_t* feats) except -1
    cdef int64_t update(self, size_t gold_i, size_t pred_i,
                        uint64_t* features, double weight) except -1
    cdef int fill_scores(self, uint64_t* features, double* scores) except -1
    cdef uint64_t predict_best_class(self, uint64_t* features)
    cdef int64_t finalize(self) except -1


