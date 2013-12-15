from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from ext.murmurhash cimport MurmurHash64A


cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    EDIT
    ASSIGN_POS
    N_MOVES


cdef FastState* init_fast_state() except NULL:
    cdef FastState* s = <FastState*>calloc(1, sizeof(FastState))
    s.knl.i = 1
    s.sig = hash_kernel(&s.knl)
    return s


cdef uint64_t hash_kernel(Kernel* k):
    return MurmurHash64A(k, sizeof(Kernel), 0)


cdef bint can_push(Kernel* k, size_t t):
    return k.i < (t-1) 


cdef bint has_stack(Kernel* k):
    return k.s0 != 0


cdef bint has_head(Kernel* k):
    return k.Ls0 != 0

cdef bint is_finished(Kernel* k, size_t t):
    return (not can_push(k, t)) and (not has_stack(k))


cdef int has_child_in_buffer(size_t word, size_t s, size_t e, size_t* heads) except -1:
    assert word != 0
    cdef size_t buff_i
    cdef int n = 0
    for buff_i in range(s, e):
        if heads[buff_i] == word:
            n += 1
    return n


cdef int has_head_in_buffer(size_t word, size_t s, size_t e, size_t* heads) except -1:
    assert word != 0
    cdef size_t buff_i
    for buff_i in range(s, e):
        if heads[word] == buff_i:
            return 1
    return 0


cdef int has_child_in_stack(size_t word, size_t length, size_t* stack, size_t* heads) except -1:
    assert word != 0
    cdef size_t i, stack_i
    cdef int n = 0
    for i in range(length):
        stack_i = stack[i]
        # Should this be sensitive to whether the word has a head already?
        if heads[stack_i] == word:
            n += 1
    return n


cdef int has_head_in_stack(size_t word, size_t length, size_t* stack, size_t* heads) except -1:
    assert word != 0
    cdef size_t i, stack_i
    for i in range(length):
        stack_i = stack[i]
        if heads[word] == stack_i:
            return 1
    return 0


cdef int shift_kernel(Kernel* result, Kernel* parent) except -1:
    result.dfl = False
    result.next_dfl = 0
    result.prev_dfl = 0
    result.i = parent.i + 1
    result.s0 = parent.i
    result.s1 = parent.s0
    result.s2 = parent.s1
    result.Ls0 = 0
    result.Ls1 = parent.Ls0
    result.Ls2 = parent.Ls1
    result.s0r.edge = result.s0
    result.n0l.edge = result.i
    # Parents of s0, e.g. hs0, h2s0, Lhs0 etc all null in Shift
    memcpy(&result.s0l, &parent.n0l, sizeof(Subtree))


cdef int right_kernel(Kernel* ext, Kernel* buff, size_t label) except -1:
    shift_kernel(ext, buff)
    ext.Ls0 = label
    ext.s0r.edge = ext.s0
    # The child-of features are set in Reduce, not here, because that's when
    # that word becomes top of the stack again.


cdef int reduce_kernel(Kernel* ext, Kernel* buff, Kernel* stack) except -1:
    memcpy(ext, stack, sizeof(Kernel))
    memcpy(&ext.n0l, &buff.n0l, sizeof(Subtree))
    ext.dfl = False
    ext.prev_dfl = buff.prev_dfl
    ext.next_dfl = stack.next_dfl
    ext.i = buff.i
    # Reduce means that former-S0 is child of the next item on the stack. Set
    # the dep features here
    ext.s0r.kids[0].idx = buff.s0
    ext.s0r.kids[0].lab = buff.Ls0
    ext.s0r.kids[1].idx = stack.s0r.kids[0].idx
    ext.s0r.kids[1].lab = stack.s0r.kids[0].lab
    ext.s0r.val = stack.s0r.val + 1
    ext.s0r.first = stack.s0r.first if ext.s0r.val >= 2 else buff.s0
    ext.s0r.edge = buff.s0r.edge


cdef int left_kernel(Kernel* ext, Kernel* buff, Kernel* stack,
                           size_t label) except -1:
    if stack != NULL:
        ext.s0 = stack.s0
        ext.s1 = stack.s1
        ext.s2 = stack.s2
        ext.Ls0 = stack.Ls0
        ext.Ls1 = stack.Ls1
        ext.Ls2 = stack.Ls2
        memcpy(&ext.s0l, &stack.s0l, sizeof(Subtree))
        memcpy(&ext.s0r, &stack.s0r, sizeof(Subtree))
    ext.dfl = False
    ext.prev_dfl = buff.prev_dfl
    ext.next_dfl = stack.next_dfl
    ext.i = buff.i
    ext.n0l.val = buff.n0l.val + 1
    ext.n0l.kids[0].idx = buff.s0
    ext.n0l.kids[0].lab = label
    ext.n0l.kids[1].idx = buff.n0l.kids[0].idx
    ext.n0l.kids[1].lab = buff.n0l.kids[0].lab
    ext.n0l.first = buff.n0l.first if ext.n0l.val >= 2 else buff.s0
    ext.n0l.edge = buff.s0l.edge


cdef int edit_kernel(Kernel* ext, Kernel* buff, Kernel* stack):
    reduce_kernel(ext, buff, stack)
    ext.dfl = True
    # This is the word immediately before i, which may be disfluent
    if buff.i == (buff.s0r.edge + 1):
        ext.prev_dfl = buff.s0r.edge
    # This is the word immediately after S0, which may be disfluent
    if (stack.s0 + 1) == buff.s0l.edge:
        ext.next_dfl = buff.s0l.edge
    # Handle the work of restoring the children to the stack in extend_fstate.
    # Here we assume stack is in the correct state for us.


cdef FastState* extend_fstate(FastState* prev, size_t move, size_t label, size_t clas,
                              double local_score, int cost) except NULL: 
    assert prev != NULL
    cdef FastState* ext = <FastState*>calloc(1, sizeof(FastState))
    if move == SHIFT:
        shift_kernel(&ext.knl, &prev.knl)
        ext.tail = prev
        ext.prev = prev
    elif move == RIGHT:
        right_kernel(&ext.knl, &prev.knl, label)
        ext.tail = prev
        ext.prev = prev
    elif move == REDUCE:
        assert prev != NULL
        assert prev.prev != NULL
        assert prev.tail != NULL
        reduce_kernel(&ext.knl, &prev.knl, &prev.tail.knl)
        ext.prev = prev
        ext.tail = prev.tail.tail
    elif move == LEFT:
        left_kernel(&ext.knl, &prev.knl, &prev.tail.knl, label)
        ext.tail = prev.tail.tail
        ext.prev = prev
    elif move == EDIT:
        _restore_lefts(ext, prev, prev.tail.tail)
        ext.prev = prev
        edit_kernel(&ext.knl, &prev.knl, &prev.tail.knl)
    else:
        raise StandardError
    assert clas < 100000
    ext.score = prev.score + local_score
    ext.cost = prev.cost + cost
    ext.clas = clas
    ext.move = move
    # The idea here is that the signature depends on the previous stack-element's
    # hash, which depends on the one behind it, etc.
    # So two states will sign the same iff they have the same local
    # hash and their tails hash the same.
    ext.sig = hash_kernel(&ext.knl)
    if ext.tail != NULL:
        ext.sig *= ext.tail.sig
    return ext


cdef int _restore_lefts(FastState* ext, FastState* buff, FastState* stack) except -1:
    cdef FastState* s
    cdef size_t s0 = buff.knl.s0
    cdef FastState* hist_state = buff
    cdef FastState* tail = ext
    while hist_state != NULL:
        if hist_state.move == LEFT and hist_state.knl.i == s0:
            s = init_fast_state()
            memcpy(&s.knl, &hist_state.prev.knl, sizeof(Kernel))
            memcpy(&s.knl.n0l, &buff.knl.n0l, sizeof(Subtree))
            s.knl.i = buff.knl.i
            # No pre-decessor state --- we shouldn't be traversing through it
            s.prev = NULL
            #s.prev = buff
            # Connect the new state to the tail
            tail.tail = s
            tail = s
        hist_state = hist_state.prev
    ext.prev = buff
    tail.tail = stack


cdef int fill_hist(size_t* hist, FastState* s, int t) except -1:
    while t >= 1 and s.prev != NULL:
        t -= 1
        hist[t] = s.clas
        s = s.prev


cdef int fill_stack(size_t* stack, FastState* s) except -1:
    cdef size_t t = 0
    while s != NULL:
        stack[t] = s.knl.s0
        s = s.tail
        t += 1
    return t - 1 if t >= 1 else 0


cdef int fill_parse(size_t* heads, size_t* labels, bint* edits, FastState* s) except -1:
    cdef size_t cnt = 0
    cdef size_t w
    while s != NULL:
        # Take the last set head, to support non-monotonicity
        # Take the heads from states just after right and left arcs
        if s.knl.Ls0 != 0 and heads[s.knl.s0] == 0:
            heads[s.knl.s0] = s.knl.s1
            labels[s.knl.s0] = s.knl.Ls0
        if s.knl.n0l.val >= 1 and heads[s.knl.n0l.kids[0].idx] == 0:
            heads[s.knl.n0l.kids[0].idx] = s.knl.i
            labels[s.knl.n0l.kids[0].idx] = s.knl.n0l.kids[0].lab
        if s.knl.dfl:
            start = s.prev.knl.s0l.edge if s.prev.knl.s0l.edge > 0 else s.prev.knl.s0
            #start = s.prev.knl.s0
            for w in range(start, s.prev.knl.s0r.edge + 1):
                heads[w] = w
                labels[w] = 5
                edits[w] = True

        s = s.prev
        cnt += 1
        assert cnt < 100000


cdef int free_fstate(FastState* s) except -1:
    cdef FastState* tmp
    while s != NULL:
        tmp = s.prev
        free(s)
        s = tmp
