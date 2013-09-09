from libc.string cimport const_void
from libc.stdint cimport uint64_t, int64_t

from ext.murmurhash cimport *


cdef struct Subtree:
    size_t val
    size_t[2] lab
    size_t[2] idx


cdef struct Kernel:
    size_t i
    size_t s0
    size_t Ls0
    size_t s1
    size_t s2
    size_t Ls1
    size_t Ls2
    Subtree s0l
    Subtree s0r
    Subtree n0l


"""
cdef struct Kernel:
    size_t i
    size_t n0p
    size_t n1p
    size_t n2p
    size_t n3p
    size_t s0
    size_t s0p
    size_t Ls0
    size_t s1
    size_t s1p
    size_t s2
    size_t s2p
    size_t Ls1
    size_t Ls2
    size_t s0ledge
    size_t s0ledgep
    size_t s0redgep
    size_t n0ledge
    size_t n0ledgep
    bint prev_edit
    bint prev_prev_edit
    bint next_edit
    bint next_next_edit
    size_t prev_tag
    size_t next_tag
    Subtree s0l
    Subtree s0r
    Subtree n0l
    size_t[5] hist
"""

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

cdef int has_child_in_buffer(size_t w, size_t s, size_t e, size_t* heads) except -1
cdef int has_head_in_buffer(size_t w, size_t s, size_t e, size_t* heads) except -1
cdef int has_child_in_stack(size_t w, size_t stack_len, size_t* stack, size_t* heads) except -1
cdef int has_head_in_stack(size_t w, size_t stack_len, size_t* stack, size_t* heads) except -1
#cdef bint has_root_child(State *s, size_t token)

cdef int fill_edits(State *s, bint* edits) except -1
cdef State* init_state(size_t n)
cdef free_state(State* s)
cdef copy_state(State* s, State* old)




#cdef int transition(self, size_t clas, State *s) except -1
#cdef int fill_static_costs(self, State* s, size_t* tags, size_t* heads,
#                           size_t* labels, bint* edits, int* costs) except -1
 
"""
cdef class Beam:
    cdef TransitionSystem trans
    #cdef priority_queue[pair[double, size_t]]* next_moves
    
    cdef State** parents
    cdef State** beam
    cdef int** valid
    cdef int** costs
    
    cdef size_t max_class
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t length
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full
    cdef bint is_finished

    cdef Kernel* next_state(self, size_t i, size_t* tags)
    cdef int extend_states(self, double** scores) except -1
    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd, bint* edits) except -1
"""


