# cython: profile=True
import io_parse
from features cimport N_LABELS

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy

DEF MAX_SENT_LEN = 200
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


cdef State* init_state(size_t n):
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
    n = n + 5
    s.stack = <size_t*>calloc(n, sizeof(size_t))
    s.heads = <size_t*>calloc(n, sizeof(size_t))
    s.labels = <size_t*>calloc(n, sizeof(size_t))
    s.guess_labels = <size_t*>calloc(n, sizeof(size_t))
    s.l_valencies = <size_t*>calloc(n, sizeof(size_t))
    s.r_valencies = <size_t*>calloc(n, sizeof(size_t))

    s.l_children = <size_t**>malloc(n * sizeof(size_t*))
    s.r_children = <size_t**>malloc(n * sizeof(size_t*))
    s.llabel_set = <bint**>malloc(n * sizeof(bint*))
    s.rlabel_set = <bint**>malloc(n * sizeof(bint*))
    cdef size_t n_labels = len(io_parse.LABEL_STRS)
    for i in range(n):
        s.l_children[i] = <size_t*>calloc(n, sizeof(size_t))
        s.r_children[i] = <size_t*>calloc(n, sizeof(size_t))
        s.llabel_set[i] = <bint*>calloc(n_labels, sizeof(bint))
        s.rlabel_set[i] = <bint*>calloc(n_labels, sizeof(bint))
    s.stack[1] = 1
    s.history = <size_t*>calloc(n * 2, sizeof(size_t))
    return s

cdef copy_state(State* s, State* old):
    cdef size_t i, j
    s.n = old.n
    s.t = old.t
    s.i = old.i
    s.score = old.score
    s.top = old.top
    s.second = old.second
    s.stack_len = old.stack_len
    s.is_finished = old.is_finished
    s.is_gold = old.is_gold
    s.at_end_of_buffer = old.at_end_of_buffer
    cdef size_t nbytes = (old.n + 5) * sizeof(size_t)
    memcpy(s.stack, old.stack, nbytes)
    memcpy(s.l_valencies, old.l_valencies, nbytes)
    memcpy(s.r_valencies, old.r_valencies, nbytes)
    memcpy(s.heads, old.heads, nbytes)
    memcpy(s.labels, old.labels, nbytes)
    memcpy(s.guess_labels, old.guess_labels, nbytes)
    memcpy(s.history, old.history, nbytes * 2)
    cdef size_t n_labels = len(io_parse.LABEL_STRS)
    for i in range(old.n + 5):
        memcpy(s.l_children[i], old.l_children[i], nbytes)
        memcpy(s.r_children[i], old.r_children[i], nbytes)
        memcpy(s.llabel_set[i], old.llabel_set[i], n_labels * sizeof(bint))
        memcpy(s.rlabel_set[i], old.rlabel_set[i], n_labels * sizeof(bint))


cdef free_state(State* s):
    free(s.stack)
    free(s.heads)
    free(s.labels)
    free(s.guess_labels)
    free(s.l_valencies)
    free(s.r_valencies)
    cdef size_t n_labels = len(io_parse.LABEL_STRS)
    for i in range(s.n + 5):
        free(s.l_children[i])
        free(s.r_children[i])
        free(s.llabel_set[i])
        free(s.rlabel_set[i])
    free(s.l_children)
    free(s.r_children)
    free(s.llabel_set)
    free(s.rlabel_set)
    free(s.history)
    free(s)



cdef int cmp_contn(const_void *c1, const_void *c2) nogil:
    cdef Cont* v1 = <Cont*>c1
    cdef Cont* v2 = <Cont*>c2
    # Reverse order
    if v1.score > v2.score:
        return -1
    elif v1.score < v2.score:
        return 1
    return 0


