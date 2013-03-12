from libc.string cimport const_void

DEF MAX_SENT_LEN = 200
DEF MAX_TRANSITIONS = MAX_SENT_LEN * 2
DEF MAX_LABELS = 50
DEF MAX_VALENCY = MAX_SENT_LEN / 2


cdef struct Subtree:
    size_t val
    size_t[4] lab
    size_t[4] idx


cdef struct Kernel:
    size_t i
    size_t s0
    size_t Ls0
    size_t hs0
    size_t h2s0
    size_t Lhs0
    size_t Lh2s0
    Subtree s0l
    Subtree s0r
    Subtree n0l


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
    size_t nr_kids
    bint is_finished
    bint at_end_of_buffer
    bint is_gold
    int cost

    size_t* stack
    size_t* heads
    size_t* labels
    size_t* guess_labels
    size_t* l_valencies
    size_t* r_valencies
    size_t** l_children
    size_t** r_children
    size_t* history
    Kernel kernel

cdef struct Cont:
    double score
    size_t clas
    size_t parent
    bint is_gold
    bint is_valid

cdef int cmp_contn(const_void* v1, const_void* v2) nogil
cdef int fill_kernel(State* s)

cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1
cdef int del_l_child(State *s, size_t head) except -1
cdef int del_r_child(State *s, size_t head) except -1

cdef size_t pop_stack(State *s) except 0
cdef int push_stack(State *s) except -1

cdef size_t get_l(State *s, size_t head)
cdef size_t get_l2(State *s, size_t head)
cdef size_t get_r(State *s, size_t head)
cdef size_t get_r2(State *s, size_t head)

cdef int has_child_in_buffer(State *s, size_t word, size_t* heads)
cdef int has_head_in_buffer(State *s, size_t word, size_t* heads)
cdef int has_child_in_stack(State *s, size_t word, size_t* heads)
cdef int has_head_in_stack(State *s, size_t word, size_t* heads)

cdef State* init_state(size_t n, size_t n_labels)
cdef free_state(State* s)
cdef copy_state(State* s, State* old, size_t n_labels)
