# cython: profile=True
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

DEF MAX_VALENCY = 100



cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1:
    s.heads[child] = head
    s.labels[child] = label
    if child < head:
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
    for i in range(2):
        tree.idx[i] = 0
        tree.lab[i] = 0
        #tree.tags[i] = 0
    tree.val = val
    if val == 0:
        return 0
    # Set 0 to be the rightmost/leftmost child, i.e. last
    tree.idx[0] = kids[val - 1]
    tree.lab[0] = labs[kids[val - 1]]
    #tree.tags[0] = tags[kids[val - 1]]
    # Set 2 to be first child
    #tree.idx[2] = kids[0]
    #tree.lab[2] = labs[kids[0]]
    #tree.tags[2] = tags[kids[0]]
    if val == 1:
        return 0
    # Set 1 to be the 2nd rightmost/leftmost, i.e. second last 
    tree.idx[1] = kids[val - 2]
    tree.lab[1] = labs[kids[val - 2]]
    #tree.tags[1] = tags[kids[val - 2]]
    # Set 3 to be second child
    #tree.idx[3] = kids[1]
    #tree.lab[3] = labs[kids[1]]
    #tree.tags[3] = tags[kids[1]]


cdef uint64_t hash_kernel(Kernel* k):
    return MurmurHash64A(k, sizeof(Kernel), 0)


cdef int fill_kernel(State *s, size_t* tags) except -1:
    cdef size_t i, val
    s.kernel.i = s.i
    s.kernel.s0 = s.top
    if s.heads[s.top] != 0 and s.heads[s.top] == s.second:
        assert s.labels[s.top] != 0
    s.kernel.s1 = s.second
    s.kernel.s2 = s.stack[s.stack_len - 3] if s.stack_len >= 3 else 0
    s.kernel.Ls0 = s.labels[s.top]
    s.kernel.Ls1 = s.labels[s.kernel.s1]
    s.kernel.Ls2 = s.labels[s.kernel.s2]
    fill_subtree(s.l_valencies[s.top], s.l_children[s.top],
                 s.labels, tags, &s.kernel.s0l)
    fill_subtree(s.r_valencies[s.top], s.r_children[s.top],
                 s.labels, tags, &s.kernel.s0r)
    fill_subtree(s.l_valencies[s.i], s.l_children[s.i],
                 s.labels, tags, &s.kernel.n0l)


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

cdef int fill_edits(State* s, bint* edits) except -1:
    cdef size_t i, j
    i = 0
    j = 0

cdef bint has_root_child(State *s, size_t token):
    # TODO: This is an SBD thingy currently not in use
    if s.at_end_of_buffer:
        return False
    return False
    # TODO: Refer to the root label constant instead here!!
    # TODO: Instead update left-arc on root so that it attaches the rest of the
    # stack to S0
    #return s.labels[get_l(s, token)] == 1


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
    s.is_finished = False
    s.at_end_of_buffer = n == 2
    n = n + PADDING
    s.stack = <size_t*>calloc(n, sizeof(size_t))
    # These make the tags match the OOB/ROOT/NONE values.
    s.heads = <size_t*>calloc(n, sizeof(size_t))
    s.labels = <size_t*>calloc(n, sizeof(size_t))
    s.guess_labels = <size_t*>calloc(n, sizeof(size_t))
    s.l_valencies = <size_t*>calloc(n, sizeof(size_t))
    s.r_valencies = <size_t*>calloc(n, sizeof(size_t))

    s.l_children = <size_t**>malloc(n * sizeof(size_t*))
    s.r_children = <size_t**>malloc(n * sizeof(size_t*))
    for i in range(n):
        s.l_children[i] = <size_t*>calloc(MAX_VALENCY, sizeof(size_t))
        s.r_children[i] = <size_t*>calloc(MAX_VALENCY, sizeof(size_t))
    s.history = <size_t*>calloc(n * 3, sizeof(size_t))
    return s

cdef copy_state(State* s, State* old):
    cdef size_t nbytes, i
    if s.i > old.i:
        nbytes = (s.i + 1) * sizeof(size_t)
    else:
        nbytes = (old.i + 1) * sizeof(size_t)
    s.n = old.n
    s.t = old.t
    s.i = old.i
    s.cost = old.cost
    s.score = old.score
    s.top = old.top
    s.second = old.second
    s.stack_len = old.stack_len
    s.is_finished = old.is_finished
    s.at_end_of_buffer = old.at_end_of_buffer
    memcpy(s.stack, old.stack, old.n * sizeof(size_t))
    memcpy(s.l_valencies, old.l_valencies, nbytes)
    memcpy(s.r_valencies, old.r_valencies, nbytes)
    memcpy(s.heads, old.heads, nbytes)
    memcpy(s.labels, old.labels, nbytes)
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
    free(s.l_valencies)
    free(s.r_valencies)
    for i in range(s.n + PADDING):
        free(s.l_children[i])
        free(s.r_children[i])
    free(s.l_children)
    free(s.r_children)
    free(s.history)
    free(s)
