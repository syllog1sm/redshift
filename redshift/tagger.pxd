from features.extractor cimport Extractor
from learn.thinc cimport LinearModel
from redshift.sentence cimport Input, Sentence, Token
from cymem.cymem cimport Pool

from libc.stdint cimport uint64_t, int64_t
from learn.thinc cimport W as weight_t


cdef class Tagger:
    cdef object cfg
    cdef Pool _pool
    cdef Extractor extractor
    cdef LinearModel guide
    cdef object model_dir
    cdef size_t beam_width

    cdef size_t* _context
    cdef uint64_t* _features
    cdef weight_t** _beam_scores

    cpdef int tag(self, Input py_sent) except -1
    cdef int train_sent(self, Input py_sent) except -1

    cdef int _predict(self, size_t i, TagState* s, Sentence* sent, weight_t* scores)


cdef class TaggerBeam:
    cdef size_t nr_class
    cdef size_t k
    cdef size_t t
    cdef size_t bsize
    cdef bint is_full
    cdef TagState** beam
    cdef TagState** parents
    cdef Pool _pool
    cdef int extend_states(self, weight_t** scores) except -1


cdef TagState* extend_state(TagState* s, size_t clas, weight_t* scores, size_t n,
                            Pool pool)

cdef int fill_hist(Token* hist, TagState* s, int t) except -1

cdef size_t get_p(TagState* s)

cdef size_t get_pp(TagState* s)

cdef struct TagState:
    weight_t score
    TagState* prev
    size_t clas
    size_t length
