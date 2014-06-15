from libc.string cimport const_void
from libc.stdint cimport uint64_t, int64_t
from libcpp.vector cimport vector

from sentence cimport Sentence, Token, Step
from transitions cimport Transition

# From left-to-right in the string, the slot tokens are:
# S2, S1, S0le, S0l, S0l2, S0l0, S0, S0r0, S0r2, S0r, S0re
# N0le, N0l, N0l2, N0l0

cdef struct SlotTokens:
    Token s2
    Token s1
    Token s1r
    Token s0le
    Token s0l
    Token s0l2
    Token s0l0
    Token s0
    Token s0r0
    Token s0r2
    Token s0r
    Token s0re
    Token n0le
    Token n0l
    Token n0l2
    Token n0l0
    Token n0
    Token n1
    Token n2

    # Previous to n0
    Token p1
    Token p2
    # After S0
    Token s0n
    Token s0nn

    # Match features
    size_t w_f_copy
    size_t w_f_exact
    size_t p_f_copy
    size_t p_f_exact
    size_t w_b_copy
    size_t w_b_exact
    size_t p_b_copy
    size_t p_b_exact

    int n0_prob


cdef struct State:
    double score
    double string_prob
    size_t i
    size_t m
    size_t n
    size_t stack_len
    size_t top
    int cost

    size_t* stack
    
    size_t** l_children
    size_t** r_children
    Token* parse
    Transition* history
    SlotTokens slots


cdef int fill_slots(State* s) except -1

cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1
cdef int del_l_child(State *s, size_t head) except -1
cdef int del_r_child(State *s, size_t head) except -1

cdef size_t pop_stack(State *s) except 0
cdef int push_stack(State *s, size_t w, Step* lattice) except -1

cdef size_t get_l(State *s, size_t head)
cdef size_t get_l2(State *s, size_t head)
cdef size_t get_r(State *s, size_t head)
cdef size_t get_r2(State *s, size_t head)

cdef size_t get_s1(State *s)

cdef bint at_eol(State *s)
cdef bint is_final(State *s)

cdef int has_child_in_buffer(State *s, size_t word, Token* gold) except -1
cdef int has_head_in_buffer(State *s, size_t word, Token* gold) except -1
cdef int has_child_in_stack(State *s, size_t word, Token* gold) except -1
cdef int has_head_in_stack(State *s, size_t word, Token* gold) except -1

cdef State* init_state(size_t length) except NULL
cdef free_state(State* s)
cdef int copy_state(State* s, State* old) except -1
