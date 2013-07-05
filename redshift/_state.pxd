from libc.string cimport const_void
from libc.stdint cimport uint64_t, int64_t


cdef extern from "MurmurHash2.h":
    uint64_t MurmurHash64A(void * key, uint64_t len, int64_t seed)
    uint64_t MurmurHash64B(void * key, uint64_t len, int64_t seed)


cdef struct Subtree:
    size_t val
    size_t[4] lab
    size_t[4] idx
    size_t[4] tags


cdef struct Kernel:
    size_t i
    size_t n0p
    size_t n1p
    size_t n2p
    size_t n3p
    size_t s0
    size_t s0p
    size_t Ls0
    size_t hs0
    size_t hs0p
    size_t h2s0
    size_t h2s0p
    size_t Lhs0
    size_t Lh2s0
    size_t s0redgep
    size_t n0ledge
    size_t n0ledgep
    bint prev_edit
    bint prev_prev_edit
    size_t prev_tag
    Subtree s0l
    Subtree s0r
    Subtree n0l


cdef struct FastState:
    Kernel* k
    size_t last_action
    FastState* previous
    FastState* tail
    double score
    bint is_gold
    size_t cost
    size_t nr_kids


cdef struct State:
    double score
    size_t i
    size_t t
    size_t n
    size_t stack_len
    size_t top
    size_t second
    bint is_finished
    bint at_end_of_buffer
    int cost

    size_t* stack
    size_t* heads
    size_t* labels
    size_t* guess_labels
    size_t* l_valencies
    size_t* r_valencies
    size_t* ledges
    size_t** l_children
    size_t** r_children
    size_t* history
    Kernel kernel

cdef uint64_t hash_kernel(Kernel* k)
cdef int fill_kernel(State* s, size_t* pos) except -1

#cdef Kernel* kernel_from_s(Kernel* parent) except NULL
#cdef Kernel* kernel_from_r(Kernel* parent, size_t label) except NULL
#cdef Kernel* kernel_from_d(Kernel* parent, Kernel* gp) except NULL
#cdef Kernel* kernel_from_l(Kernel* parent, Kernel* gp, size_t label) except NULL

cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1
cdef int del_l_child(State *s, size_t head) except -1
cdef int del_r_child(State *s, size_t head) except -1

cdef size_t pop_stack(State *s) except 0
cdef int push_stack(State *s) except -1

cdef size_t get_l(State *s, size_t head)
cdef size_t get_l2(State *s, size_t head)
cdef size_t get_r(State *s, size_t head)
cdef size_t get_r2(State *s, size_t head)

cdef int has_child_in_buffer(State *s, size_t word, size_t* heads) except -1
cdef int has_head_in_buffer(State *s, size_t word, size_t* heads) except -1
cdef int has_child_in_stack(State *s, size_t word, size_t* heads) except -1
cdef int has_head_in_stack(State *s, size_t word, size_t* heads) except -1
cdef bint has_root_child(State *s, size_t token)

cdef int fill_edits(State *s, bint* edits) except -1
cdef State* init_state(size_t n)
cdef free_state(State* s)
cdef copy_state(State* s, State* old)
