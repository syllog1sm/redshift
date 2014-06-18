from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
from redshift.sentence cimport Input, Sentence, Token, Step
from index.lexicon cimport Lexeme
from ext.sparsehash cimport dense_hash_map

from libc.stdint cimport uint64_t, int64_t


cdef struct Slots:
    size_t pp_tag
    size_t p_tag
    Lexeme* pp_word
    Lexeme* p_word
    Lexeme* n0
    Lexeme* n1


cdef class Tagger:
    cdef object cfg
    cdef Extractor extractor
    cdef Perceptron guide
    cdef object model_dir
    cdef dense_hash_map[uint64_t, size_t] tagdict

    cdef Slots slots
    cdef bint _cache_hit
    cdef size_t* _context
    cdef uint64_t* _features
    cdef size_t _guessed

    cdef int tag_word(self, Token* state, size_t i, Step* lattice, size_t n) except -1
    cdef int tell_gold(self, size_t gold) except -1
