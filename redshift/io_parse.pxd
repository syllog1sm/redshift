

cdef struct _Parse:
    size_t n_moves
    size_t* heads
    size_t* labels
    bint* sbd
    size_t* moves

cdef struct Sentence:
    size_t id
    size_t length
    size_t* ids
    size_t* words
    size_t* owords
    size_t* pos
    size_t* clusters
    size_t* cprefix4s
    size_t* cprefix6s
    size_t* orths
    size_t* parens
    size_t* quotes
    _Parse* parse


cdef Sentence* make_sentence(size_t id_, size_t length, object py_ids,
                            object py_words, object py_tags, size_t vocab_thresh)

cdef free_sent(Sentence* s)

cdef class Sentences:
    cdef object strings
    cdef Sentence** s
    cdef size_t length
    cdef size_t vocab_thresh 
    cdef size_t max_length

    cpdef int add(self, size_t id_, object ids, object words, object tags,
                  object heads, object labels) except -1
