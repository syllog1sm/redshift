DEF MAX_SENT_LEN = 300
DEF MAX_TRANSITIONS = MAX_SENT_LEN * 2
DEF MAX_LABELS = 50
DEF MAX_VALENCY = MAX_SENT_LEN / 2


# With low MAX_SENT_LEN most of these can be reduced to char instead of size_t,
# but what's the point? We usually only have one state at a time
cdef struct State:
    double score
    size_t i
    size_t t
    size_t n
    size_t stack_len
    size_t top
    size_t second
    size_t[MAX_SENT_LEN] stack
    size_t[MAX_SENT_LEN] heads
    size_t[MAX_SENT_LEN] labels
    size_t guess_labels[MAX_SENT_LEN][MAX_SENT_LEN]
    size_t[MAX_SENT_LEN] l_valencies
    size_t[MAX_SENT_LEN] r_valencies
    size_t l_children[MAX_SENT_LEN][MAX_VALENCY]
    size_t r_children[MAX_SENT_LEN][MAX_VALENCY]
    bint llabel_set[MAX_SENT_LEN][MAX_LABELS]
    bint rlabel_set[MAX_SENT_LEN][MAX_LABELS]
    size_t[MAX_TRANSITIONS] history
    bint is_finished
    bint at_end_of_buffer


cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1
cdef int del_l_child(State *s, size_t head) except -1
cdef int del_r_child(State *s, size_t head) except -1

cdef size_t pop_stack(State *s) except 0
cdef int push_stack(State *s) except -1

cdef int get_l(State *s, size_t head) except -1
cdef int get_l2(State *s, size_t head) except -1
cdef int get_r(State *s, size_t head) except -1
cdef int get_r2(State *s, size_t head) except -1

cdef int get_left_edge(State *s, size_t head) except -1
cdef int get_right_edge(State *s, size_t head) except -1

cdef bint has_child_in_buffer(State *s, size_t word, size_t* heads)
cdef bint has_head_in_buffer(State *s, size_t word, size_t* heads)
cdef bint has_child_in_stack(State *s, size_t word, size_t* heads)
cdef bint has_head_in_stack(State *s, size_t word, size_t* heads)
cdef bint has_head_via_lower(State *s, size_t word, size_t* heads)
cdef bint has_grandchild_via_lower(State *s, size_t word, size_t* heads)

cdef State init_state(size_t n)
