from libcpp.vector cimport vector
from libcpp.utility cimport pair

from libc.stdint cimport int64_t, uint64_t

DEF MAX_PARAMS = 5000000

DEF MAX_DENSE = 100000

#cdef extern from "predict.h":
cdef struct Param:
    double w
    double acc
    size_t clas
    size_t last_upd

cdef struct Feature:
    Param** params
    int* index
    size_t n_upd
    size_t n_class
    size_t max_class


cdef void update_param(Feature* feat, uint64_t clas, uint64_t now, double weight)
cdef void update_dense(size_t now, size_t nr_class, uint64_t f, uint64_t clas,
                       double weight, double* w, double* acc, size_t* last_upd)


cdef class MultitronParameters:
    cdef uint64_t n_classes
    cdef uint64_t n_params
    cdef uint64_t max_classes
    cdef uint64_t max_param
    cdef uint64_t max_dense
    cdef uint64_t true_nr_class
    cdef uint64_t now
    cdef double* w
    cdef double* acc
    cdef size_t* last_upd
    cdef Feature** W
    cdef bint* seen
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


