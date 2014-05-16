from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
from redshift.sentence cimport Input, Sentence, Token
from ext.sparsehash cimport dense_hash_map

from libc.stdint cimport uint64_t, int64_t


cdef class Tagger:
    cdef object cfg
    cdef Extractor extractor
    cdef Perceptron guide
    cdef object model_dir
    cdef size_t beam_width
    cdef dense_hash_map[size_t, size_t] tagdict

    cdef size_t* _context
    cdef uint64_t* _features
    cdef double** beam_scores
    cpdef int tag(self, Input py_sent) except -1
    cdef int train_sent(self, Input py_sent) except -1

    cdef int _predict(self, size_t i, TagState* s, Sentence* sent, double* scores)


cdef class TaggerBeam:
    cdef size_t nr_class
    cdef size_t k
    cdef size_t t
    cdef size_t bsize
    cdef bint is_full
    cdef set seen_states
    cdef TagState** beam
    cdef TagState** parents
    cdef int extend_states(self, double** scores) except -1


cdef TagState* extend_state(TagState* s, size_t clas, double* scores, size_t n)

cdef int fill_hist(Token* hist, TagState* s, int t) except -1

cdef size_t get_p(TagState* s)

cdef size_t get_pp(TagState* s)

cdef struct TagState:
    double score
    TagState* prev
    size_t clas
    size_t length
