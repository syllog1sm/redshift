cimport index.hashes
from libcpp.vector cimport vector
from libcpp.utility cimport pair

from libc.stdint cimport int64_t, uint64_t
from ext.sparsehash cimport *

DEF MAX_PARAMS = 5000000

DEF MAX_DENSE = 100000


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
    cdef DenseFeature** _active_dense
    cdef SquareFeature** _active_square
    cdef object path
    cdef bint is_trained
    cdef float n_corr
    cdef float total
    cdef bint accept_new_feats

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
    cdef int unfinalize(self) except -1
