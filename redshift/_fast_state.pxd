from _state cimport Kernel

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

cdef Kernel* shift_kernel(Kernel* result, Kernel* parent) except NULL
cdef Kernel* right_kernel(Kernel* result, Kernel* parent, size_t label) except NULL
cdef Kernel* reduce_kernel(Kernel* result, Kernel* parent, Kernel* gp) except NULL
cdef Kernel* left_kernel(Kernel* result, Kernel* parent,
                          Kernel* gp, size_t label) except NULL


cdef FastState* extend_fstate(FastState* prev, size_t move, size_t label,
                              size_t clas, double local_score, int cost) except NULL 


cdef int fill_hist(size_t* hist, FastState* s, int t) except -1
