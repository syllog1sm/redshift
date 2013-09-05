
cdef struct Subtree:
    size_t val
    size_t[4] lab
    size_t[4] idx
    size_t[4] tags


cdef struct LKernel:
    size_t i
    size_t s0
    size_t Ls0
    size_t s1
    size_t s2
    size_t Ls1
    size_t Ls2
    size_t s0ledge
    size_t n0ledge
    Subtree s0l
    Subtree s0r
    Subtree n0l


cdef struct FastState:
    LKernel* knl
    size_t clas
    FastState* prev
    double score
    size_t cost


cdef LKernel* shift_kernel(LKernel* result, LKernel* parent) except NULL
cdef LKernel* right_kernel(LKernel* result, LKernel* parent, size_t label) except NULL
cdef LKernel* reduce_kernel(LKernel* result, LKernel* parent, LKernel* gp) except NULL
cdef LKernel* left_kernel(LKernel* result, LKernel* parent,
                         LKernel* gp, size_t label) except NULL


cdef FastState* extend_fstate(FastState* prev, size_t move, size_t label,
                              size_t clas, double local_score, int cost) except NULL 


cdef int fill_hist(size_t* hist, FastState* s, int t) except -1
