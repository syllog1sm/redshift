# cython: profile=True
import io_parse
from features cimport N_LABELS

from libc.stdlib cimport malloc, free

DEF MAX_SENT_LEN = 300
DEF MAX_TRANSITIONS = MAX_SENT_LEN * 2
DEF MAX_VALENCY = MAX_SENT_LEN / 2


cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1:
    s.heads[child] = head
    s.labels[child] = label
    if child < head:
        assert s.l_valencies[head] < MAX_VALENCY
        assert s.l_children[head][s.l_valencies[head]] == 0
        s.l_children[head][s.l_valencies[head]] = child
        s.l_valencies[head] += 1
        s.llabel_set[head][label] = 1
    else:
        assert s.r_valencies[head] < MAX_VALENCY, s.r_valencies[head]
        r = get_r(s, head)
        if r != 0:
            assert r < child, r
        s.r_children[head][s.r_valencies[head]] = child
        s.r_valencies[head] += 1
        s.rlabel_set[head][label] = 1
    return 1

cdef int del_r_child(State *s, size_t head) except -1:
    child = get_r(s, head)
    assert s.r_valencies[head] >= 1
    assert child > 0
    s.r_children[head][s.r_valencies[head] - 1] = 0
    s.r_valencies[head] -= 1
    old_label = s.labels[child]
    for i in range(s.r_valencies[head]):
        if s.labels[s.r_children[head][i]] == old_label:
            break
    else:
        s.rlabel_set[head][old_label] = 0
    s.heads[child] = 0
    s.labels[child] = 0

cdef int del_l_child(State *s, size_t head) except -1:
    cdef:
        size_t i
        size_t child
        size_t old_label
    assert s.l_valencies[head] >= 1
    child = get_l(s, head)
    s.l_children[head][s.l_valencies[head] - 1] = 0
    s.l_valencies[head] -= 1
    old_label = s.labels[child]
    for i in range(s.l_valencies[head]):
        if s.labels[s.l_children[head][i]] == old_label:
            break
    else:
        s.llabel_set[head][old_label] = 0
    s.heads[child] = 0
    s.labels[child] = 0

cdef size_t pop_stack(State *s) except 0:
    cdef size_t popped
    assert s.stack_len > 1
    popped = s.top
    s.stack_len -= 1
    s.top = s.second
    if s.stack_len >= 2:
        s.second = s.stack[s.stack_len - 2]
    else:
        s.second = 0
    assert s.top <= s.n, s.top
    assert popped != 0
    return popped


cdef int push_stack(State *s) except -1:
    s.second = s.top
    s.top = s.i
    s.stack[s.stack_len] = s.i
    s.stack_len += 1
    assert s.top <= s.n
    s.i += 1

cdef int get_l(State *s, size_t head) except -1:
    if s.l_valencies[head] == 0:
        return 0
    return s.l_children[head][s.l_valencies[head] - 1]

cdef int get_l2(State *s, size_t head) except -1:
    if s.l_valencies[head] < 2:
        return 0
    return s.l_children[head][s.l_valencies[head] - 2]

cdef int get_r(State *s, size_t head) except -1:
    if s.r_valencies[head] == 0:
        return 0
    return s.r_children[head][s.r_valencies[head] - 1]

cdef int get_r2(State *s, size_t head) except -1:
    if s.r_valencies[head] < 2:
        return 0
    return s.r_children[head][s.r_valencies[head] - 2]


cdef int get_left_edge(State *s, size_t node) except -1:
    if s.l_valencies[node] == 0:
        return 0
    node = s.l_children[node][s.l_valencies[node] - 1]
    while s.l_valencies[node] != 0:
        node = s.l_children[node][s.l_valencies[node] - 1]
    return node

cdef int get_right_edge(State *s, size_t node) except -1:
    if s.r_valencies[node] == 0:
        return 0
    node = s.r_children[node][s.r_valencies[node] - 1]
    while s.r_valencies[node] != 0:
        node = s.r_children[node][s.r_valencies[node] - 1]
    return node

cdef bint has_child_in_buffer(State *s, size_t word, size_t* heads):
    assert word != 0
    cdef size_t buff_i
    for buff_i in range(s.i, s.n):
        if heads[buff_i] == word:
            return True
    return False

cdef bint has_head_in_buffer(State *s, size_t word, size_t* heads):
    assert word != 0
    cdef size_t buff_i
    for buff_i in range(s.i, s.n):
        if heads[word] == buff_i:
            return True
    return False

cdef bint has_child_in_stack(State *s, size_t word, size_t* heads):
    assert word != 0
    cdef size_t i, stack_i
    for i in range(1, s.stack_len):
        stack_i = s.stack[i]
        # Should this be sensitie to whether the word has a head already?
        if heads[stack_i] == word:
            return True
    return False

cdef bint has_head_in_stack(State *s, size_t word, size_t* heads):
    assert word != 0
    cdef size_t i, stack_i
    for i in range(1, s.stack_len):
        stack_i = s.stack[i]
        if heads[word] == stack_i:
            return True
    return False

cdef bint has_head_via_lower(State *s, size_t word, size_t* heads):
    return False
    # Check whether the head is recoverable via the Lower transition
    # Assumes the transition's enabled.
    #for i in range(1, s.stack_len):
    #    stack_i = s.stack[i]
    #    if get_r(s, stack_i) != 0 and heads[word] == get_r(s, stack_i):
    #        return True
        # TODO: Should this be updated for r2 being a head?
    #return False

cdef bint has_grandchild_via_lower(State *s, size_t word, size_t* heads):
    # Check whether we need to keep the stack item around for the sake of 
    # the children: i.e., being able to attach a word and Lower it onto the
    # grandchild.
    #r = get_r(s, word)
    #if r == 0:
    #    return False
    #for buff_i in range(s.i, s.n - 1):
    #    if heads[buff_i] == r:
    #        return True
    return False


cdef State* init_state(size_t n):
    # TODO: Make this more efficient, probably by storing 0'd arrays somewhere,
    # and then copying them
    cdef size_t i, j
    cdef State* s = <State*>malloc(sizeof(State))

    s.n = n
    s.t = 0
    s.i = 2
    s.score = 0
    s.top = 1
    s.second = 0
    s.stack_len = 2
    s.is_finished = False
    s.is_gold = True
    s.at_end_of_buffer = n == 3
    s.stack = <size_t*>malloc(n * sizeof(size_t))
    s.heads =  <size_t*>malloc(n * sizeof(size_t))
    s.labels =  <size_t*>malloc(n * sizeof(size_t))
    s.l_valencies = <size_t*>malloc(n * sizeof(size_t))
    s.r_valencies = <size_t*>malloc(n * sizeof(size_t))
    s.guess_labels = <size_t**>malloc(n * sizeof(size_t*))
    s.l_children = <size_t**>malloc(n * sizeof(size_t*))
    s.r_children = <size_t**>malloc(n * sizeof(size_t*))
    s.llabel_set = <bint**>malloc(n * sizeof(bint*))
    s.rlabel_set = <bint**>malloc(n * sizeof(bint*))
    s.history = <size_t*>malloc(n * 2 * sizeof(size_t))
    cdef size_t n_labels = len(io_parse.LABEL_STRS)
    for i in range(n):
        s.stack[i] = 0
        s.l_valencies[i] = 0
        s.r_valencies[i] = 0
        s.heads[i] = 0 
        s.labels[i] = 0
        s.guess_labels[i] = <size_t*>malloc(n * sizeof(size_t))
        s.l_children[i] = <size_t*>malloc(n * sizeof(size_t))
        s.r_children[i] = <size_t*>malloc(n * sizeof(size_t))
        # Ideally this shouldn't matter, if we use valencies intelligently?
        for j in range(n):
            s.guess_labels[i][j] = 0
            s.l_children[i][j] = 0
            s.r_children[i][j] = 0
        s.llabel_set[i] = <bint*>malloc(n_labels * sizeof(bint))
        s.rlabel_set[i] = <bint*>malloc(n_labels * sizeof(bint))
        for j in range(n_labels):
            s.llabel_set[i][j] = False
            s.rlabel_set[i][j] = False
    s.stack[1] = 1
    for i in range(2 * n):
        s.history[i] = 0
    return s


cdef State* copy_state(State* old):
    cdef size_t i, j
    cdef State* s = <State*>malloc(sizeof(State))
    cdef size_t n = old.n
    s.stack = <size_t*>malloc(n * sizeof(size_t))
    s.heads =  <size_t*>malloc(n * sizeof(size_t))
    s.labels =  <size_t*>malloc(n * sizeof(size_t))
    s.l_valencies = <size_t*>malloc(n * sizeof(size_t))
    s.r_valencies = <size_t*>malloc(n * sizeof(size_t))
    s.guess_labels = <size_t**>malloc(n * sizeof(size_t*))
    s.l_children = <size_t**>malloc(n * sizeof(size_t*))
    s.r_children = <size_t**>malloc(n * sizeof(size_t*))
    s.llabel_set = <bint**>malloc(n * sizeof(bint*))
    s.rlabel_set = <bint**>malloc(n * sizeof(bint*))
    s.history = <size_t*>malloc(n * 2 * sizeof(size_t))

    s.n = old.n
    s.t = old.t
    s.i = old.i
    s.top = old.top
    s.second = old.second
    s.stack_len = old.stack_len
    s.is_finished = old.is_finished
    s.at_end_of_buffer = old.at_end_of_buffer
    s.is_gold = old.is_gold
    s.score = old.score
    cdef size_t n_labels = len(io_parse.LABEL_STRS)
    for i in range(old.n):
        s.stack[i] = old.stack[i]
        s.l_valencies[i] = old.l_valencies[i]
        s.r_valencies[i] = old.r_valencies[i]
        s.heads[i] = old.heads[i]
        s.labels[i] = old.labels[i]
        s.guess_labels[i] = <size_t*>malloc(s.n * sizeof(size_t))
        s.l_children[i] = <size_t*>malloc(s.n * sizeof(size_t))
        s.r_children[i] = <size_t*>malloc(s.n * sizeof(size_t))
        # Ideally this shouldn't matter, if we use valencies intelligently?
        for j in range(s.n):
            s.guess_labels[i][j] = old.guess_labels[i][j]
            s.l_children[i][j] = old.l_children[i][j]
            s.r_children[i][j] = old.r_children[i][j]
        s.llabel_set[i] = <bint*>malloc(n_labels * sizeof(bint))
        s.rlabel_set[i] = <bint*>malloc(n_labels * sizeof(bint))
        for j in range(n_labels):
            s.llabel_set[i][j] = old.llabel_set[i][j]
            s.rlabel_set[i][j] = old.rlabel_set[i][j]
    for i in range(2 * s.n):
        s.history[i] = old.history[i]
    return s

cdef int free_state(State* s):
    cdef size_t i
    free(s.stack)
    free(s.heads)
    free(s.labels)
    free(s.l_valencies)
    free(s.r_valencies)
    free(s.history)
    for i in range(s.n):
        free(s.guess_labels[i])
        free(s.l_children[i])
        free(s.r_children[i])
        free(s.llabel_set[i])
        free(s.rlabel_set[i])
    free(s.guess_labels)
    free(s.l_children)
    free(s.r_children)
    free(s.llabel_set)
    free(s.rlabel_set)
    free(s)


cdef int cmp_contn(const void *c1, const void *c2):
    cdef double v1 = c1.score
    cdef double v2 = c2.score
    if v1 < v2:
        return -1
    elif v1 > v2:
        return 1
    return 0


