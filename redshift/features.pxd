from libc.stdint cimport uint64_t, int64_t
from _state cimport State, Kernel
from io_parse cimport Sentence
from libcpp.vector cimport vector
from libcpp.utility cimport pair

from ext.murmurhash cimport *

cdef struct Predicate:
    int id, n, expected_size
    uint64_t* raws
    int* args


cdef struct MatchPred:
    size_t id
    size_t idx1
    size_t idx2
    size_t[2] raws


cdef class FeatureSet:
    cdef object name
    cdef object ngrams
    cdef bint add_clusters
    cdef uint64_t mask_value
    cdef Predicate** predicates
    cdef MatchPred** match_preds
    cdef size_t* context
    cdef uint64_t* features 
    cdef int n
    cdef int nr_match
    cdef int nr_label
    cdef uint64_t* extract(self, Sentence* sent, Kernel* k) except NULL
