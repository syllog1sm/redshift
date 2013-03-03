from libc.stdint cimport uint64_t
from _state cimport State
from io_parse cimport Sentence
from index.hashes cimport FeatIndex

cdef struct Predicate:
    int id, n, expected_size
    uint64_t* raws
    int* args

cdef class FeatureSet:
    cdef Predicate* predicates
    cdef size_t* context
    cdef uint64_t* features 
    cdef FeatIndex feat_idx
    cdef int n
    cdef int nr_label
    cdef uint64_t* extract(self, Sentence* sent, State* state) except NULL

    cdef int _make_predicates(self, bint add_extra) except 0
    


cdef int CONTEXT_SIZE


cdef int fill_context(size_t* context, size_t nr_label, size_t n0, size_t n1, size_t n2,
                      size_t, size_t s1,
                      size_t s0_re, size_t s1_re,
                      size_t stack_len,
                      size_t* words, size_t* pos, size_t* browns,
                      size_t* heads, size_t* labels, size_t* l_vals, size_t* r_vals,
                      size_t* s0_lkids, size_t* s0_rkids, size_t* s1_lkids, size_t* s1_rkids,
                      size_t* n0_lkids,
                      bint* s0_llabels, bint* s0_rlabels, bint* n0_llabels) except -1

