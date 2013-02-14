from libcpp.vector cimport vector
from libcpp.utility cimport pair

from libc.stdint cimport int64_t, uint64_t

DEF MAX_PARAMS = 20000000

cdef struct ParamData:
    double* w
    double* acc
    size_t* last_upd
    size_t n_upd


cdef class MultitronParameters:
    cdef uint64_t n_classes
    cdef uint64_t n_params
    cdef uint64_t max_classes
    cdef uint64_t max_param
    cdef uint64_t true_nr_class
    cdef uint64_t now
    cdef ParamData** W
    cdef int64_t* feat_idx
    cdef double* scores
    cdef uint64_t* labels
    cdef int64_t* label_to_i
    
    cdef tick(self)
    cdef int64_t lookup_label(self, uint64_t label) except -1
    cdef int64_t add_param(self, uint64_t f)
    cdef int64_t prune_rares(self, size_t thresh)
    cdef int64_t update(self, uint64_t gold_label, uint64_t pred_label,
                        uint64_t n_feats, uint64_t* features, double weight) except -1
    cdef double* get_scores(self, uint64_t n_feats, uint64_t* features)
    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features)
    cdef int64_t finalize(self) except -1


