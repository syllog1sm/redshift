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


cdef uint64_t hash_kernel(Kernel* k):
    return MurmurHash64A(k, sizeof(Kernel), 0)


cdef int fill_kernel(State *s, size_t* tags) except -1:
    cdef size_t i, val
    cdef size_t* slots = s.kernel.slots

    assert s.ledges[s.i] != 0
    s.kernel.i = s.i
    s.kernel.segment = s.segment

    # S2, S1 
    slots[0] = s.stack[s.stack_len - 3] if s.stack_len >= 3 else 0
    slots[1] = s.second

    # S0le, S0l, S0l2, S0l0
    slots[2] = s.ledges[s.top]
    slots[3] = get_l(s, s.top)
    slots[4] = get_l2(s, s.top)
    slots[5] = s.l_children[s.top][0]

    # S0
    slots[6] = s.top

    # S0r0, S0r2, S0r, S0re
    slots[7] = s.r_children[s.top][0]
    slots[8] = get_r2(s, s.top)
    slots[9] = get_r(s, s.top)
    slots[10] = s.ledges[s.i] - 1 # IE S0re is the word before N0le

    # N0le, N0l, N0l2, N0l0
    slots[11] = s.ledges[s.i]
    slots[12] = get_l(s, s.i)
    slots[13] = get_l2(s, s.i)
    slots[14] = s.l_children[s.i][0]
    # N0
    slots[15] = s.i

    for i in range(14):
        s.kernel.tags[i] = s.tags[slots[i]]
        s.kernel.labels[i] = s.labels[slots[i]]
        s.kernel.l_vals[i] = s.l_valencies[slots[i]]
        s.kernel.r_vals[i] = s.r_valencies[slots[i]]
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
            # TODO: Test this bug fix
            s.kernel.prev_prev_edit = False
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
    else:
        # TODO: Test this bug fix
        s.kernel.next_edit = False
        s.kernel.next_next_edit = False


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
