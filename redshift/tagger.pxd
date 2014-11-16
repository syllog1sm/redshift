from thinc.features cimport Extractor
from thinc.learner cimport LinearModel
from redshift.sentence cimport Input, Sentence, Token
from cymem.cymem cimport Pool

from thinc.search cimport Beam, MaxViolation
from thinc.typedefs cimport atom_t, feat_t, weight_t



cdef class Tagger:
    cdef object cfg
    cdef Pool _pool
    cdef Extractor extractor
    cdef LinearModel guide
    cdef object model_dir
    cdef size_t beam_width

    cdef atom_t* _context
    cdef feat_t* _features
    cdef weight_t* _values
    cdef weight_t** _beam_scores

    cpdef int tag(self, Input py_sent) except -1
    cdef int train_sent(self, Input py_sent) except -1

    cdef int _predict(self, size_t i, TagState* s, Sentence* sent, weight_t* scores)
    cdef dict _count_feats(self, Sentence* sent, TagState* p, TagState* g, int i)


cdef TagState* extend_state(TagState* s, size_t clas, weight_t score, size_t cost,
                            Pool pool)

cdef inline size_t get_p(TagState* s) nogil

cdef inline size_t get_pp(TagState* s) nogil


cdef struct TagState:
    weight_t score
    int cost
    TagState* prev
    size_t clas
    size_t length
