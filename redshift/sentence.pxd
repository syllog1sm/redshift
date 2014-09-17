from index.lexicon cimport Lexeme
from memsafe cimport Pool


cdef struct Token:
    size_t i
    Lexeme* word # Supports confusion network
    size_t tag
    size_t head
    size_t label
    size_t left_edge
    size_t l_valency
    size_t r_valency
    size_t sent_id
    bint is_edit


cdef struct Step:
    size_t n
    double* probs
    Lexeme** nodes


cdef struct Sentence:
    size_t n
    Step* lattice
    Token* tokens
    double score
    

cdef Sentence* init_sent(list words_lattice, list parse, Pool pool) except NULL

cdef class Input:
    cdef Pool _pool
    cdef Sentence* c_sent
