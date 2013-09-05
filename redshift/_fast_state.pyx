from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy

cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    EDIT
    ASSIGN_POS
    N_MOVES


cdef LKernel* shift_kernel(LKernel* result, LKernel* parent) except NULL:
    result.i = parent.i + 1
    result.s2 = parent.s1
    result.Ls2 = parent.Ls1
    result.s1 = parent.s0
    result.Ls1 = parent.Ls0
    result.s0 = parent.i
    result.Ls0 = 0
    result.s0ledge = result.n0ledge
    result.n0ledge = result.i
    # Parents of s0, e.g. hs0, h2s0, Lhs0 etc all null in Shift
    memcpy(&result.s0l, &parent.n0l, sizeof(Subtree))
    return result 


cdef LKernel* right_kernel(LKernel* result, LKernel* parent, size_t label) except NULL:
    shift_kernel(result, parent)
    result.s0ledge = parent.s0ledge
    result.n0ledge = result.i
    result.Ls0 = label
    return result


cdef LKernel* reduce_kernel(LKernel* result, LKernel* parent, LKernel* grandparent) except NULL:
    memcpy(result, grandparent, sizeof(LKernel))
    memcpy(&result.n0l, &parent.n0l, sizeof(Subtree))
    result.i = parent.i
    return result


cdef LKernel* left_kernel(LKernel* result, LKernel* parent, LKernel* grandparent,
                           size_t label) except NULL:
    assert parent.s0 >= grandparent.s0
    memcpy(result, grandparent, sizeof(LKernel))
    result.i = parent.i
    result.n0ledge = parent.s0ledge
    result.s0ledge = grandparent.s0ledge
    result.n0l.val = parent.n0l.val + 1
    result.n0l.idx[0] = parent.s0
    result.n0l.idx[1] = parent.n0l.idx[0]
    result.n0l.idx[2] = 0
    result.n0l.idx[3] = 0
    result.n0l.lab[0] = label
    result.n0l.lab[1] = parent.n0l.idx[0]
    result.n0l.lab[2] = parent.n0l.idx[1]
    result.n0l.lab[3] = parent.n0l.idx[2]
    return result


cdef FastState* extend_fstate(FastState* prev, size_t move, size_t label, size_t clas,
                              double local_score, int cost) except NULL: 
    assert prev != NULL
    cdef FastState* ext = <FastState*>calloc(1, sizeof(FastState))
    if move == SHIFT:
        shift_kernel(ext.knl, prev.knl)
    elif move == REDUCE:
        assert prev.prev != NULL
        reduce_kernel(ext.knl, prev.knl, prev.prev.knl)
    elif move == RIGHT:
        right_kernel(ext.knl, prev.knl, label)
    elif move == LEFT:
        assert prev.prev != NULL
        left_kernel(ext.knl, prev.knl, prev.prev.knl, label)
    else:
        raise StandardError

    ext.score = prev.score + local_score
    ext.cost = prev.cost + cost
    ext.clas = clas
    return ext


cdef int fill_hist(size_t* hist, FastState* s, int t) except -1:
    while t >= 1 and s.prev != NULL:
        t -= 1
        hist[t] = s.clas
        s = s.prev
