from index.vocab cimport Token

cdef struct AnswerToken:
    size_t word # Supports confusion network
    size_t tag
    size_t head
    size_t label
    bint is_break
    bint is_edit

cdef struct Step:
    size_t n
    double* probs
    Token** nodes

cdef struct Sentence:
    size_t n
    Step* steps
    AnswerToken* answer
    double score
    
cdef Sentence* init_sent(size_t id_, size_t length, py_words)

cdef free_sent(Sentence* s)

cdef class PySentence:
    cdef size_t id
    cdef Sentence* c_sent
    cdef size_t length
    cdef list sausage
