

cdef struct _Parse:
    size_t n_moves
    size_t* heads
    size_t* labels
    size_t* moves
    size_t* move_labels

cdef struct Sentence:
    size_t id
    size_t length
    size_t* words
    size_t* pos
    size_t* browns
    _Parse parse


cdef Sentence make_sentence(size_t id_, size_t length, object py_words, object py_tags)

cdef class Sentences:
    cdef object strings
    cdef Sentence *s
    cdef size_t length
    cdef size_t max_length

    cpdef int add(self, size_t id_, object words, object tags, object heads, object labels) except -1

    cdef evaluate(self, Sentences gold)
