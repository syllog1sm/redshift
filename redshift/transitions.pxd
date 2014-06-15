from _state cimport State, SlotTokens

from redshift.sentence cimport Token
from redshift.sentence cimport Step

from libcpp.vector cimport vector


cdef struct Transition:
    size_t clas
    size_t move
    size_t label
    double score
    int cost
    bint is_valid

cdef object move_name(Transition* t)

cdef size_t get_nr_moves(size_t shift_classes, size_t lattice_width, list left_labels,
                         list right_labels, list dfl_labels, bint use_break)

cdef int fill_moves(size_t shift_classes, size_t lattice_width, list left_labels,
                    list right_labels, list dfl_labels, bint use_break, Transition* moves)

cdef int fill_valid(State* s, Step* lattice, Transition* classes, size_t n) except -1

cdef int fill_costs(State* s, Step* lattice, Transition* classes, size_t n, Token* gold) except -1

cdef int transition(Transition* t, State *s, Step* lattice) except -1
