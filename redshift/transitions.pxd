from _state cimport State, SlotTokens

from redshift.sentence cimport Token

from libcpp.vector cimport vector



cdef struct Transition:
    size_t clas
    size_t move
    size_t label
    double score
    int cost
    bint is_valid

cdef size_t get_nr_moves(list left_labels, list right_labels, list dfl_labels,
                         bint use_break)

cdef int fill_moves(list left_labels, list right_labels, list dfl_labels,
                    bint use_break, Transition* moves)

cdef int fill_valid(State* s, Transition* classes, size_t n) except -1

cdef int fill_costs(State* s, Transition* classes, size_t n, Token* gold) except -1

cdef int transition(Transition* t, State *s) except -1

cdef int transition_slots(SlotTokens* slots, Transition* t) except -1

cdef int add_valid_moves(vector[Transition] moves, State* s, bint force_gold) except -1


