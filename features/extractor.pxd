from libc.stdint cimport uint64_t, int64_t

from ext.murmurhash cimport *

cdef struct Template:
    size_t id
    size_t n
    uint64_t* raws
    int* args


cdef struct MatchPred:
    size_t id
    size_t idx1
    size_t idx2
    size_t[2] raws


cdef class Extractor:
    cdef size_t nr_template
    cdef Template** templates
    cdef size_t nr_match
    cdef size_t nr_bow
    cdef size_t nr_feat
    cdef size_t* for_bow
    cdef MatchPred** match_preds
    cdef int extract(self, uint64_t* features, size_t* context) except -1
