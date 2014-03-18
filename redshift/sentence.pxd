from index.vocab cimport Word

cdef struct AnswerToken:
    size_t word # Supports confusion network
    size_t tag
    size_t head
    size_t label
    size_t left_edge
    size_t l_valency
    size_t r_valency
    bint is_break
    bint is_edit

cdef struct Step:
    size_t n
    double* probs
    Word** nodes

cdef struct Sentence:
    size_t n
    Step* steps
    AnswerToken* answer
    double score
    
cdef Sentence* init_sent(list words_cn) except NULL

cdef void free_sent(Sentence* s)

cdef class Input:
    cdef Sentence* c_sent


cdef class Token:
    cdef Word* c_word
    cdef AnswerToken* c_parse
