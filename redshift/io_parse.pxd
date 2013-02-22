from libcpp.vector cimport vector

cimport _state

DEF MAX_SENT_LEN = 500
DEF MAX_TRANSITIONS = (500 * 2) - 2


cdef struct _Parse:
    size_t n_moves
    size_t[MAX_SENT_LEN] heads
    size_t[MAX_SENT_LEN] labels
    size_t[MAX_TRANSITIONS] moves
    size_t[MAX_TRANSITIONS] move_labels

cdef struct Sentence:
    size_t id
    size_t length
    size_t[MAX_SENT_LEN] words
    size_t[MAX_SENT_LEN] pos
    size_t[MAX_SENT_LEN] browns
    _Parse parse


cdef Sentence make_sentence(size_t id_, size_t length, object py_words, object py_tags)

cdef class Sentences:
    cdef object strings
    cdef Sentence *s
    cdef size_t length
    cdef size_t max_length

    cpdef int add(self, size_t id_, object words, object tags, object heads, object labels) except -1

    cdef evaluate(self, Sentences gold)
