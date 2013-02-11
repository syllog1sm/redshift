from libcpp.vector cimport vector
from libcpp.utility cimport pair

from numpy cimport int64_t

#cdef class MulticlassParamData:
#    cdef double *acc
#    cdef double *w
#    cdef int *lastUpd

cdef extern from "sparsehash/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        size_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        size_t count(K&)
        iterator begin()
        iterator end()
        size_t size()
        size_t max_size()
        bint empty()
        size_t bucket_count()
        size_t bucket_size(size_t i)
        size_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(size_t n)
        void rehash(size_t n)
        dense_hash_map()
        dense_hash_map(size_t n)


        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        size_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil



cdef struct ParamData:
    double* acc
    int* lastUpd


cdef class MultitronParameters:
    cdef size_t n_classes
    cdef size_t n_params
    cdef size_t max_classes
    cdef size_t max_param
    cdef size_t true_nr_class
    cdef size_t now
    #cdef dense_hash_map[size_t, ParamData] W
    cdef vector[ParamData] W
    cdef vector[vector[double]] weights
    cdef vector[int64_t] feat_idx
    cdef double* scores
    cdef size_t* labels
    cdef int* label_to_i
    
    cdef tick(self)
    cdef int lookup_label(self, size_t label) except -1
    cdef int add_param(self, size_t f)
    cdef int update(self, size_t gold_label, size_t pred_label, size_t n_feats, size_t* features) except -1
    cdef double* get_scores(self, size_t n_feats, size_t* features)
    cdef size_t predict_best_class(self, size_t n_feats, size_t* features)
    cdef int finalize(self) except -1


    #cpdef pa_update(self, object gu_feats, object go_feats, int gu_cls, int go_cls,double C=?)
    #cpdef add_params(self, MultitronParameters other, double factor)
    #cpdef set(self, list features, int clas, double amount)
    #cpdef do_pa_update(self, list feats, int gold_cls, double C=?)
    #cpdef get_scores_r(self, features)
    #cpdef add_r(self, list features, int clas, double amount)
    #cpdef scalar_multiply(self, double scalar)
    #cdef _update(self, int goodClass, int badClass, list features)

    #cdef _update_r(self, int goodClass, int badClass, list features)

    #cpdef getW(self, object clas)

    #cpdef predict_best_class_r(self, list features)
