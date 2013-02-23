# cython: profile=True
import io_parse
from features cimport N_LABELS

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

DEF START_ON_STACK = True

cdef State init_state(size_t n):
    # TODO: Make this more efficient, probably by storing 0'd arrays somewhere,
    # and then copying them
    cdef size_t i, j
    cdef State s
    cdef int n_labels = len(io_parse.LABEL_STRS)
    # Initialise with first word on top of stack
    assert n >= 3
    if START_ON_STACK:
        s = State(n=n, t=0, i=2, top=1, second=0, stack_len=2, is_finished=False,
                  at_end_of_buffer=n == 3)
    else:
        s = State(n=n, t=0, i=1, top=0, second=0, stack_len=1, is_finished=False,
                  at_end_of_buffer=n == 3)
    for i in range(n):
        s.stack[i] = 0
        s.l_valencies[i] = 0
        s.r_valencies[i] = 0
        s.heads[i] = 0 
        s.labels[i] = 0
        # Ideally this shouldn't matter, if we use valencies intelligently?
        for j in range(n):
            s.guess_labels[i][j] = 0
            s.l_children[i][j] = 0
            s.r_children[i][j] = 0
        for j in range(n_labels):
            s.llabel_set[i][j] = 0
            s.rlabel_set[i][j] = 0
    if START_ON_STACK:
        s.stack[1] = 1
    for i in range(MAX_TRANSITIONS):
        s.history[i] = 0
    return s
