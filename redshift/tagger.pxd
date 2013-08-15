from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
from redshift.io_parse cimport Sentence, Sentences
from redshift.beam cimport TagState, TaggerBeam

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
 
