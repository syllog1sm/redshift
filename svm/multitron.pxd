from libcpp.vector cimport vector
from libcpp.utility cimport pair

from numpy cimport int64_t, uint64_t

cdef struct ParamData:
    double* acc
    uint64_t* lastUpd


cdef class MultitronParameters:
    cdef uint64_t n_classes
    cdef uint64_t n_params
    cdef uint64_t max_classes
    cdef uint64_t max_param
    cdef uint64_t true_nr_class
    cdef uint64_t now
    #cdef dense_hash_map[size_t, ParamData] W
    cdef vector[ParamData] W
    cdef vector[vector[double]] weights
    cdef vector[int64_t] feat_idx
    cdef double* scores
    cdef uint64_t* labels
    cdef int64_t* label_to_i
    
    cdef tick(self)
    cdef int64_t lookup_label(self, uint64_t label) except -1
    cdef int64_t add_param(self, uint64_t f)
    cdef int64_t update(self, uint64_t gold_label, uint64_t pred_label, uint64_t n_feats, uint64_t* features) except -1
    cdef double* get_scores(self, uint64_t n_feats, uint64_t* features)
    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features)
    cdef int64_t finalize(self) except -1


