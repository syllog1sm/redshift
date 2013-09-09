from libc.stdint cimport uint64_t, int64_t
from _state cimport Subtree, Kernel

#cdef struct Subtree:
#    size_t val
#    size_t[2] lab
#    size_t[2] idx


#cdef struct Kernel:
#    size_t i
#    size_t s0
#    size_t Ls0
#    size_t s1
#    size_t s2
#    size_t Ls1
#    size_t Ls2
#    Subtree s0l
#    Subtree s0r
#    Subtree n0l


cdef struct FastState:
    Kernel knl
    size_t clas
    FastState* prev
    FastState* tail
    double score
    size_t cost

cdef FastState* init_fast_state() except NULL

cdef bint can_push(Kernel* k, size_t t)
cdef bint has_stack(Kernel* k)
cdef bint has_head(Kernel* k)
cdef bint is_finished(Kernel* k, size_t length)

cdef uint64_t hash_kernel(Kernel* k)

cdef int shift_kernel(Kernel* result, Kernel* parent) except -1
cdef int right_kernel(Kernel* result, Kernel* parent, size_t label) except -1
cdef int reduce_kernel(Kernel* result, Kernel* parent, Kernel* gp) except -1
cdef int left_kernel(Kernel* result, Kernel* parent, Kernel* gp, size_t label) except -1


cdef FastState* extend_fstate(FastState* prev, size_t move, size_t label,
                              size_t clas, double local_score, int cost) except NULL 


cdef int fill_hist(size_t* hist, FastState* s, int t) except -1

cdef int fill_stack(size_t* stack, FastState* s) except -1

cdef int free_fstate(FastState* s) except -1

