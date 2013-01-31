
from _state cimport State
from io_parse cimport Sentence

cdef struct Predicate:
    int id, n, expected_size
    size_t* raws
    int* args

cdef int PAD_SIZE

cdef int N_PREDICATES

cdef int CONTEXT_SIZE
cdef int N_LABELS


cdef int make_predicates(bint add_labels, bint add_extra) except 0

cdef Predicate* predicates

cdef size_t* init_context()

cdef size_t* init_hashed_features()

cdef int fill_context(size_t* context, size_t n0, size_t n1, n2,
                      size_t s0, size_t s1, size_t s2, size_t s3,
                      size_t s0_re, size_t s1_re,
                      size_t stack_len,
                      size_t* words, size_t* pos, size_t* browns,
                      size_t* heads, size_t* labels, size_t* l_vals, size_t* r_vals,
                      size_t* s0_lkids, size_t* s0_rkids, size_t* s1_lkids, size_t* s1_rkids,
                      size_t* n0_lkids,
                      bint* s0_llabels, bint* s0_rlabels, bint* n0_llabels) except -1

cdef int extract(size_t* context, size_t* hashed, Sentence* sent,
        State* state) except -1

cdef set_n_labels(int n)
