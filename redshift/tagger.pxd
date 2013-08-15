from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
from redshift.io_parse cimport Sentence, Sentences

from libc.stdint cimport uint64_t, int64_t


cdef class BaseTagger:
    cdef Extractor features
    cdef Perceptron guide
    cdef object model_dir
    cdef size_t beam_width
    cdef int feat_thresh
    cdef size_t max_feats
    cdef size_t nr_tag
    cdef size_t _acc

    cdef size_t* _context
    cdef uint64_t* _features
    cdef double** beam_scores

    
    cdef int tag(self, Sentence* s) except -1

    cdef int train_sent(self, Sentence* sent) except -1

cdef class GreedyTagger(BaseTagger):
    pass


cdef class BeamTagger(BaseTagger):
    cdef TagState* extend_gold(self, TagState* s, Sentence* sent, size_t i) except NULL
    cdef int fill_beam_scores(self, TaggerBeam beam, Sentence* sent,
                              size_t word_i) except -1
 


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

cdef int fill_hist(size_t* hist, TagState* s, int t) except -1

cdef size_t get_p(TagState* s)

cdef size_t get_pp(TagState* s)

cdef struct TagState:
    double score
    TagState* prev
    size_t alt
    size_t clas
    size_t length
