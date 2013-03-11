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

cdef struct Feature:
    Param** params
    int* index
    size_t n_class
    size_t max_class


cdef struct DenseParams:
    double* w
    double* acc
    size_t* last_upd


cdef struct DenseFeature:
    DenseParams* parts    


cdef void update_param(Feature* feat, uint64_t clas, uint64_t now, double weight)
cdef void update_dense(size_t now, size_t nr_class, uint64_t f, uint64_t clas,
                       double weight, double* w, double* acc, size_t* last_upd)


cdef class MultitronParameters:
    cdef size_t nr_class
    cdef size_t nr_label
    cdef uint64_t now
    cdef dense_hash_map[uint64_t, size_t] W
    cdef double* scores
    cdef uint64_t* labels
    cdef int64_t* label_to_i
    
    cdef tick(self)
    cdef int64_t lookup_label(self, uint64_t label) except -1
    cdef int64_t add_feature(self, uint64_t f)
    cdef int64_t prune_rares(self, size_t thresh)
    cdef int64_t update(self, uint64_t gold_label, uint64_t pred_label,
                        uint64_t n_feats, uint64_t* features, double weight) except -1

    cdef int update_single(self, uint64_t label, uint64_t f, double weight) except -1
    cdef int get_scores(self, uint64_t n_feats, uint64_t* features, double* scores) except -1
    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features)
    cdef int64_t finalize(self) except -1


