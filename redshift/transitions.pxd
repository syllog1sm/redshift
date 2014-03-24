from _state cimport State 

from redshift.sentence cimport Token


cdef struct Transition:
    size_t clas
    size_t move
    size_t label
    double score
    int cost
    bint is_valid

cdef size_t get_nr_moves(list left_labels, list right_labels)

cdef int fill_moves(list left_labels, list right_labels, Transition* moves)

cdef int fill_valid(State* s, Transition* classes, size_t n) except -1

cdef int fill_costs(State* s, Transition* classes, size_t n, Token* gold) except -1

cdef int transition(Transition* t, State *s) except -1


cdef int right_cost(State *s, Token* gold)

cdef bint can_right(State* s)
