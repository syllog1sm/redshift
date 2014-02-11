# cython: profile=True
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

DEF MAX_VALENCY = 100

cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1:
    s.heads[child] = head
    s.labels[child] = label
    if child < head:
        s.ledges[head] = s.ledges[child]
        if s.l_valencies[head] < MAX_VALENCY:
            s.l_children[head][s.l_valencies[head]] = child
            s.l_valencies[head] += 1
    else:
        if s.r_valencies[head] < MAX_VALENCY:
            s.r_children[head][s.r_valencies[head]] = child
            s.r_valencies[head] += 1


cdef int del_r_child(State *s, size_t head) except -1:
    cdef size_t child = get_r(s, head)
    s.r_children[head][s.r_valencies[head] - 1] = 0
    s.r_valencies[head] -= 1
    s.heads[child] = 0
    s.labels[child] = 0


cdef int del_l_child(State *s, size_t head) except -1:
    cdef:
        size_t i
        size_t child
        size_t old_label
    child = get_l(s, head)
    s.l_children[head][s.l_valencies[head] - 1] = 0
    s.l_valencies[head] -= 1
    s.heads[child] = 0
    s.labels[child] = 0
    # This assertion ensures the left-edge above stays correct.
    assert s.heads[head] == 0 or s.heads[head] <= head
    if s.l_valencies[head] != 0:
        s.ledges[head] = s.ledges[get_l(s, head)]
    else:
        s.ledges[head] = head


cdef size_t pop_stack(State *s) except 0:
    cdef size_t popped
    assert s.stack_len >= 1
    popped = s.top
    s.stack_len -= 1
    s.top = s.second
    if s.stack_len >= 2:
        s.second = s.stack[s.stack_len - 2]
    else:
        s.second = 0
    assert s.top <= s.n, s.top
    assert popped != 0
    cdef size_t child
    if s.stack_len == 0:
        s.segment = False
    return popped


cdef int push_stack(State *s) except -1:
    s.second = s.top
    s.top = s.i
    s.stack[s.stack_len] = s.i
    s.stack_len += 1
    assert s.top <= s.n
    s.i += 1


cdef int fill_subtree(size_t val, size_t* kids, size_t* labs, size_t* tags,  Subtree* tree):
    cdef size_t i
    for i in range(4):
        tree.idx[i] = 0
        tree.lab[i] = 0
        tree.tags[i] = 0
    tree.val = val
    if val == 0:
        return 0
    # Set 0 to be the rightmost/leftmost child, i.e. last
    tree.idx[0] = kids[val - 1]
    tree.lab[0] = labs[kids[val - 1]]
    tree.tags[0] = tags[kids[val - 1]]
    # Set 2 to be first child
    tree.idx[2] = kids[0]
    tree.lab[2] = labs[kids[0]]
    tree.tags[2] = tags[kids[0]]
    if val == 1:
        return 0
    # Set 1 to be the 2nd rightmost/leftmost, i.e. second last 
    tree.idx[1] = kids[val - 2]
    tree.lab[1] = labs[kids[val - 2]]
    tree.tags[1] = tags[kids[val - 2]]
    # Set 3 to be second child
    tree.idx[3] = kids[1]
    tree.lab[3] = labs[kids[1]]
    tree.tags[3] = tags[kids[1]]


cdef uint64_t hash_kernel(Kernel* k):
    return MurmurHash64A(k, sizeof(Kernel), 0)


cdef int fill_kernel(State *s, size_t* tags) except -1:
    cdef size_t i, val
    s.kernel.segment = s.segment
    s.kernel.i = s.i
    s.kernel.n0p = tags[s.i]
    s.kernel.n1p = tags[s.i + 1]
    s.kernel.n2p = tags[s.i + 2]
    s.kernel.n3p = tags[s.i + 3]
    s.kernel.s0 = s.top
    if s.heads[s.top] != 0 and s.heads[s.top] == s.second:
        assert s.labels[s.top] != 0
    s.kernel.s0p = tags[s.top]
    s.kernel.s1 = s.second
    s.kernel.s1p = tags[s.kernel.s1]
    s.kernel.s2 = s.stack[s.stack_len - 3] if s.stack_len >= 3 else 0
    s.kernel.s2p = tags[s.kernel.s2]
    s.kernel.Ls0 = s.labels[s.top]
    s.kernel.Ls1 = s.labels[s.kernel.s1]
    s.kernel.Ls2 = s.labels[s.kernel.s2]
    s.kernel.s0ledge = s.ledges[s.top]
    s.kernel.s0ledgep = tags[s.ledges[s.top]]
    s.kernel.n0ledge = s.ledges[s.i]
    s.kernel.n0ledgep = tags[s.ledges[s.i]]
    if s.ledges[s.i] != 0:
        s.kernel.s0redgep = tags[s.ledges[s.i] - 1]
    else:
        s.kernel.s0redgep = 0
    cdef size_t prev
    cdef size_t prev_prev
    if s.i > 0:
        prev = s.i - 1
        s.kernel.prev_edit = True if s.heads[prev] == prev else False
        s.kernel.prev_tag = tags[prev]
        if s.i > 1:
            prev_prev = s.i - 2
            s.kernel.prev_prev_edit = True if s.heads[prev_prev] == prev_prev else False
    else:
        s.kernel.prev_edit = False
        s.kernel.prev_prev_edit = False
        s.kernel.prev_tag = False
    cdef size_t next_
    cdef size_t next_next
    if s.top != 0:
        next_ = s.top + 1
        s.kernel.next_edit = True if s.heads[next_] == next_ else False
        s.kernel.next_tag = tags[next_]
        next_next = s.top + 2
        s.kernel.next_next_edit = True if s.heads[next_next] == next_next else False
    fill_subtree(s.l_valencies[s.top], s.l_children[s.top],
                 s.labels, tags, &s.kernel.s0l)
    fill_subtree(s.r_valencies[s.top], s.r_children[s.top],
                 s.labels, tags, &s.kernel.s0r)
    fill_subtree(s.l_valencies[s.i], s.l_children[s.i],
                 s.labels, tags, &s.kernel.n0l)
    if s.t >= 5:
        s.kernel.hist[0] = s.history[s.t - 1]
        s.kernel.hist[1] = s.history[s.t - 2]
        s.kernel.hist[2] = s.history[s.t - 3]
        s.kernel.hist[3] = s.history[s.t - 4]
        s.kernel.hist[4] = s.history[s.t - 5]
    else:
        for i in range(s.t):
            s.kernel.hist[i] = s.history[s.t - (i + 1)]
        for i in range(s.t, 5):
            s.kernel.hist[i] = 0


#cdef Kernel* kernel_from_s(Kernel* parent) except NULL:
#    k = <Kernel*>malloc(sizeof(Kernel))
#    memset(k, 0, sizeof(Kernel))
#    k.i = parent.i + 1
#    k.s0 = parent.i
#    k.s0ledge = parent.n0ledge
#    k.n0ledge = k.i
#    # Parents of s0, e.g. hs0, h2s0, Lhs0 etc all null in Shift
#    memcpy(&k.s0l, &parent.n0l, sizeof(Subtree))
#    return k


#cdef Kernel* kernel_from_r(Kernel* parent, size_t label) except NULL:
#    cdef Kernel* k = kernel_from_s(parent)
#    k.s0ledge = parent.s0ledge
#    k.n0ledge = k.i
#    k.Ls0 = label
#    k.hs0 = parent.s0
#    k.h2s0 = parent.hs0
#    k.Lhs0 = parent.Ls0
#    k.Lh2s0 = parent.Lhs0
#    return k


#cdef Kernel* kernel_from_d(Kernel* parent, Kernel* grandparent) except NULL:
#    assert parent.s0 >= grandparent.s0
#    k = <Kernel*>malloc(sizeof(Kernel))
#    memcpy(k, grandparent, sizeof(Kernel))
#    memcpy(&k.n0l, &parent.n0l, sizeof(Subtree))
#    k.i = parent.i
#    return k


#cdef Kernel* kernel_from_l(Kernel* parent, Kernel* grandparent, size_t label) except NULL:
#    assert parent.s0 >= grandparent.s0
#    k = <Kernel*>malloc(sizeof(Kernel))
#    memcpy(k, grandparent, sizeof(Kernel))
#    k.i = parent.i
#    k.n0ledge = parent.s0ledge
#    k.s0ledge = grandparent.s0ledge
#    k.n0l.val = parent.n0l.val + 1
#    k.n0l.idx[0] = parent.s0
#    k.n0l.idx[1] = parent.n0l.idx[0]
#    k.n0l.idx[2] = 0
#    k.n0l.idx[3] = 0
#    k.n0l.lab[0] = label
#    k.n0l.lab[1] = parent.n0l.idx[0]
#    k.n0l.lab[2] = parent.n0l.idx[1]
#    k.n0l.lab[3] = parent.n0l.idx[2]
#    return k

cdef size_t get_l(State *s, size_t head):
    if s.l_valencies[head] == 0:
        return 0
    return s.l_children[head][s.l_valencies[head] - 1]

cdef size_t get_l2(State *s, size_t head):
    if s.l_valencies[head] < 2:
        return 0
    return s.l_children[head][s.l_valencies[head] - 2]

cdef size_t get_r(State *s, size_t head):
    if s.r_valencies[head] == 0:
        return 0
    return s.r_children[head][s.r_valencies[head] - 1]

cdef size_t get_r2(State *s, size_t head):
    if s.r_valencies[head] < 2:
        return 0
    return s.r_children[head][s.r_valencies[head] - 2]

cdef int has_child_in_buffer(State *s, size_t word, size_t* heads) except -1:
    assert word != 0
    cdef size_t buff_i
    cdef int n = 0
    for buff_i in range(s.i, s.n):
        if heads[buff_i] == word:
            n += 1
    return n

cdef int has_head_in_buffer(State *s, size_t word, size_t* heads) except -1:
    assert word != 0
    cdef size_t buff_i
    for buff_i in range(s.i, s.n):
        if heads[word] == buff_i:
            return 1
    return 0

cdef int has_child_in_stack(State *s, size_t word, size_t* heads) except -1:
    assert word != 0
    cdef size_t i, stack_i
    cdef int n = 0
    for i in range(s.stack_len):
        stack_i = s.stack[i]
        # Should this be sensitie to whether the word has a head already?
        if heads[stack_i] == word:
            n += 1
    return n


cdef int has_head_in_stack(State *s, size_t word, size_t* heads) except -1:
    assert word != 0
    cdef size_t i, stack_i
    for i in range(s.stack_len):
        stack_i = s.stack[i]
        if heads[word] == stack_i:
            return 1
    return 0

cdef int nr_headless(State* s) except -1:
    cdef size_t n = 0
    cdef size_t i
    for i in range(s.stack_len):
        n += s.heads[s.stack[i]] == 0
    return n

cdef int fill_edits(State* s, bint* edits) except -1:
    cdef size_t i, j
    i = 0
    j = 0
    while i <= s.n:
        if i != 0 and s.heads[i] == i:
            edits[i] = True
            start = s.ledges[i]
            end = i
            while s.r_valencies[end] != 0:
                end = get_r(s, end)
            end += 1
            #print "Editing %d-%d" % (start, end)
            for k in range(start, end):
                edits[k] = True
            i = end
        else:
            i += 1


cdef bint has_root_child(State *s, size_t token):
    if s.at_end_of_buffer:
        return False
    # TODO: Refer to the root label constant instead here!!
    # TODO: Instead update left-arc on root so that it attaches the rest of the
    # stack to S0
    return s.labels[get_l(s, token)] == 3


DEF PADDING = 5


cdef State* init_state(size_t n):
    cdef size_t i, j
    cdef State* s = <State*>calloc(1, sizeof(State))
    s.n = n
    s.t = 0
    s.i = 1
    s.cost = 0
    s.score = 0
    s.top = 0
    s.second = 0
    s.stack_len = 0
    s.segment = False
    s.is_finished = False
    s.at_end_of_buffer = n == 2
    n = n + PADDING
    s.stack = <size_t*>calloc(n, sizeof(size_t))
    # These make the tags match the OOB/ROOT/NONE values.
    s.heads = <size_t*>calloc(n, sizeof(size_t))
    s.labels = <size_t*>calloc(n, sizeof(size_t))
    s.sbd = <size_t*>calloc(n, sizeof(size_t))
    s.guess_labels = <size_t*>calloc(n, sizeof(size_t))
    s.l_valencies = <size_t*>calloc(n, sizeof(size_t))
    s.r_valencies = <size_t*>calloc(n, sizeof(size_t))

    s.l_children = <size_t**>malloc(n * sizeof(size_t*))
    s.r_children = <size_t**>malloc(n * sizeof(size_t*))
    s.ledges = <size_t*>malloc(n * sizeof(size_t))
    for i in range(n):
        s.ledges[i] = i
        s.l_children[i] = <size_t*>calloc(MAX_VALENCY, sizeof(size_t))
        s.r_children[i] = <size_t*>calloc(MAX_VALENCY, sizeof(size_t))
    s.history = <size_t*>calloc(n * 3, sizeof(size_t))
    return s

cdef int copy_state(State* s, State* old) except -1:
    cdef size_t nbytes, i
    if s.i > old.i:
        nbytes = (s.i + 1) * sizeof(size_t)
    else:
        nbytes = (old.i + 1) * sizeof(size_t)
    s.n = old.n
    s.t = old.t
    s.i = old.i
    s.segment = old.segment
    s.cost = old.cost
    s.score = old.score
    s.top = old.top
    s.second = old.second
    s.stack_len = old.stack_len
    s.is_finished = old.is_finished
    s.at_end_of_buffer = old.at_end_of_buffer
    memcpy(s.stack, old.stack, old.n * sizeof(size_t))
    memcpy(s.ledges, old.ledges, (old.n + PADDING) * sizeof(size_t))
    memcpy(s.l_valencies, old.l_valencies, nbytes)
    memcpy(s.r_valencies, old.r_valencies, nbytes)
    memcpy(s.heads, old.heads, nbytes)
    memcpy(s.labels, old.labels, nbytes)
    memcpy(s.sbd, old.sbd, nbytes)
    memcpy(s.guess_labels, old.guess_labels, nbytes)
    memcpy(s.history, old.history, old.t * sizeof(size_t))
    for i in range(old.i + 2):
        memcpy(s.l_children[i], old.l_children[i], old.l_valencies[i] * sizeof(size_t))
        memcpy(s.r_children[i], old.r_children[i], old.r_valencies[i] * sizeof(size_t))


cdef free_state(State* s):
    free(s.stack)
    free(s.heads)
    free(s.labels)
    free(s.guess_labels)
    free(s.sbd)
    free(s.l_valencies)
    free(s.r_valencies)
    free(s.ledges)
    for i in range(s.n + PADDING):
        free(s.l_children[i])
        free(s.r_children[i])
    free(s.l_children)
    free(s.r_children)
    free(s.history)
    free(s)
