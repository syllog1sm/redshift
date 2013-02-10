from libcpp.vector cimport vector
from libcpp.utility cimport pair

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
        iterator begin()
        iterator end()
        size_t size()
        size_t max_size()
        bint empty()
        size_t bucket_count()
        size_t bucket_size(size_t i)
        size_t bucket(K& key)
        size_t count(K& count)
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
    double* w
    double* acc
    int* lastUpd
    int* class_to_i
    size_t* non_zeroes
    size_t n_non_zeroes


cdef class MultitronParameters:
    cdef size_t n_classes
    cdef size_t max_classes
    cdef size_t max_param
    cdef size_t now
    cdef int tick(self)
    cdef dense_hash_map[size_t, ParamData] W
    cdef double* scores
    cdef size_t* labels
    cdef int* label_to_i
    cdef double* _double_zeroes
    
    cdef int lookup_label(self, size_t label) except -1
    cdef int lookup_class(self, ParamData* p, size_t clas) except -1
    cdef int add_param(self, size_t f)
    cdef int update(self, size_t gold_label, size_t pred_label, size_t n_feats, size_t* features) except -1
    cdef double* get_scores(self, size_t n_feats, size_t* features)
    cdef size_t predict_best_class(self, size_t n_feats, size_t* features)
    cdef int finalize(self) except -1
