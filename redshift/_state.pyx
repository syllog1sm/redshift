# cython: profile=True
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

from redshift.sentence cimport Step
from index.lexicon cimport BLANK_WORD, get_str

from libc.math cimport floor

DEF MAX_VALENCY = 200

cdef int add_dep(State *s, size_t head, size_t child, size_t label) except -1:
    s.parse[child].head = head
    s.parse[child].label = label
    if child < head:
        s.parse[head].left_edge = s.parse[child].left_edge
        if s.parse[head].l_valency < MAX_VALENCY:
            s.l_children[head][s.parse[head].l_valency] = child
            s.parse[head].l_valency += 1
    elif s.parse[head].r_valency < MAX_VALENCY:
        s.r_children[head][s.parse[head].r_valency] = child
        s.parse[head].r_valency += 1


cdef int del_r_child(State *s, size_t head) except -1:
    cdef size_t child = get_r(s, head)
    s.r_children[head][s.parse[head].r_valency - 1] = 0
    s.parse[head].r_valency -= 1
    s.parse[child].head = 0
    s.parse[child].label = 0


cdef int del_l_child(State *s, size_t head) except -1:
    cdef size_t child = get_l(s, head)
    s.l_children[head][s.parse[head].l_valency - 1] = 0
    s.parse[head].l_valency -= 1
    s.parse[child].head = 0
    s.parse[child].label = 0
    # This assertion ensures the left-edge above stays correct.
    assert s.parse[head].head == 0 or s.parse[head].head <= head
    if s.parse[head].l_valency != 0:
        s.parse[head].left_edge = s.parse[get_l(s, head)].left_edge
    else:
        s.parse[head].left_edge = head


cdef size_t pop_stack(State *s) except 0:
    cdef size_t popped
    assert s.stack_len >= 1
    popped = s.top
    s.top = get_s1(s)
    s.stack_len -= 1
    assert s.top <= s.n, s.top
    assert popped != 0
    return popped


cdef int push_stack(State *s, size_t w, Step* lattice) except -1:
    if s.i != 0:
        s.top = s.i
        s.stack[s.stack_len] = s.i
        s.stack_len += 1
    assert s.top <= s.n
    s.i += 1
    s.parse[s.i].word = lattice[s.i].nodes[w]
    assert s.parse[s.i].word != NULL
    s.parse[s.i].sent_id = s.parse[s.top].sent_id # TODO: Do we need this?


cdef int fill_slots(State *s) except -1:
    s.slots.s2 = s.parse[s.stack[s.stack_len - 3] if s.stack_len >= 3 else 0]
    s.slots.s1 = s.parse[get_s1(s)]
    s.slots.s1r = s.parse[get_r(s, get_s1(s))]
    s.slots.s0le = s.parse[s.parse[s.top].left_edge]
    s.slots.s0l = s.parse[get_l(s, s.top)]
    s.slots.s0l2 = s.parse[get_l2(s, s.top)]
    s.slots.s0l0 = s.parse[s.l_children[s.top][0]]
    s.slots.s0 = s.parse[s.top]
    s.slots.s0r = s.parse[get_r(s, s.top)]
    s.slots.s0r2 = s.parse[get_r2(s, s.top)]
    s.slots.s0r0 = s.parse[s.r_children[s.top][0]]
    # IE S0re is the word before N0le
    if s.i == 0:
        s.slots.s0re = s.parse[0]
    else:
        assert s.parse[s.i].left_edge != 0
        s.slots.s0re = s.parse[s.parse[s.i].left_edge - 1]
    s.slots.n0le = s.parse[s.parse[s.i].left_edge]
    s.slots.n0l = s.parse[get_l(s, s.i)]
    s.slots.n0l2 = s.parse[get_l2(s, s.i)]
    s.slots.n0l0 = s.parse[s.l_children[s.i][0]]
    s.slots.n0 = s.parse[s.i]
    s.slots.n1 = s.parse[s.i + 1 if s.n >= 1 and  s.i < (s.n - 1) else 0]
    s.slots.n2 = s.parse[s.i + 2 if s.n >= 2 and s.i < (s.n - 2) else 0]
    s.slots.s0n = s.parse[s.top + 1 if s.top and s.n >= 1 and s.top < (s.n - 1) else 0]
    s.slots.s0nn = s.parse[s.top + 2 if s.top and s.n >= 2 and s.top < (s.n - 2) else 0]

    cdef int i = s.i
    i = i-1 if i >= 1 else 0
    while i >= 1 and s.parse[i].is_edit:
        i -= 1
    s.slots.p1 = s.parse[i]
    i = i-1 if i >= 1 else 0
    while i >= 1 and s.parse[i].is_edit:
        i -= 1
    s.slots.p2 = s.parse[i]
    i = i-1 if i >= 1 else 0
    while i >= 1 and s.parse[i].is_edit:
        i -= 1
    s.slots.p3 = s.parse[i]
    # Force to int values between 0 and -10
    if s.string_prob < -10:
        s.slots.n0_prob = -10
    elif s.string_prob == 1:
        s.slots.n0_prob = 1
    elif s.string_prob != 0:
        s.slots.n0_prob = <int>floor(s.string_prob)
    else:
        s.slots.n0_prob = 0

    s.slots.w_f_exact = 0
    s.slots.w_b_exact = 0
    s.slots.p_f_exact = 0
    s.slots.p_b_exact = 0
    s.slots.w_f_copy = 0
    s.slots.w_b_copy = 0
    s.slots.p_f_copy = 0
    s.slots.p_b_copy = 0
    """
    # Match features
    cdef size_t[5] s1
    cdef size_t[5] s2
    cdef size_t len1 = _fill_words(s1, s.top, 5, s.i, s.parse)
    cdef size_t len2 = _fill_words(s2, s.i, 5, s.n, s.parse)
    s.slots.w_f_exact = _fill_match(&s.slots.w_f_copy, s1, s2, min(len1, len2))

    len1 = _fill_words(s1, s.top, -5, 0, s.parse)
    len2 = _fill_words(s2, s.i, -5, s.top, s.parse)
    s.slots.w_b_exact = _fill_match(&s.slots.w_b_copy, s1, s2, min(len1, len2))

    len1 = _fill_tags(s1, s.top, 5, s.i, s.parse)
    len2 = _fill_tags(s2, s.i, 5, s.n, s.parse)
    s.slots.p_f_exact = _fill_match(&s.slots.p_f_copy, s1, s2, min(len1, len2))

    len1 = _fill_tags(s1, s.top, -5, 0, s.parse)
    len2 = _fill_tags(s2, s.i, -5, s.top, s.parse)
    s.slots.p_b_exact = _fill_match(&s.slots.p_b_copy, s1, s2, min(len1, len2))
    """
 
cdef size_t _fill_words(size_t* string, size_t start, int n, size_t stop, Token* tokens):
    cdef size_t i = 0
    if n >= 0:
        for i in range(min(n, stop - start)):
            string[i] = <size_t>tokens[start + i].word.orig
    else:
        for i in range(min(-n, start - stop)):
            string[i] = <size_t>tokens[start - i].word.orig
    return i

cdef size_t _fill_tags(size_t* string, size_t start, int n, size_t stop, Token* tokens):
    cdef size_t i = 0
    if n >= 0:
        for i in range(min(n, stop - start)):
            string[i] = <size_t>tokens[start + i].tag
    else:
        for i in range(min(-n, start - stop)):
            string[i] = <size_t>tokens[start - i].tag
    return i

cdef bint _fill_match(size_t* match_len, size_t* s1, size_t* s2, size_t n):
    cdef size_t i = 0
    for i in range(n):
        if s1[i] == s2[i]:
            match_len[0] += 1
        else:
            return False
    return i != 0
 

cdef size_t get_s1(State *s):
    if s.stack_len < 2:
        return 0
    return s.stack[s.stack_len - 2]

cdef size_t get_l(State *s, size_t head):
    if s.parse[head].l_valency == 0:
        return 0
    return s.l_children[head][s.parse[head].l_valency - 1]

cdef size_t get_l2(State *s, size_t head):
    if s.parse[head].l_valency < 2:
        return 0
    return s.l_children[head][s.parse[head].l_valency - 2]

cdef size_t get_r(State *s, size_t head):
    if s.parse[head].r_valency == 0:
        return 0
    return s.r_children[head][s.parse[head].r_valency - 1]

cdef size_t get_r2(State *s, size_t head):
    if s.parse[head].r_valency < 2:
        return 0
    return s.r_children[head][s.parse[head].r_valency - 2]

cdef int has_child_in_buffer(State *s, size_t word, Token* gold) except -1:
    assert word != 0
    cdef size_t buff_i
    # For efficiency, don't search past the word's head, if its head is to the right
    cdef size_t stop = gold[word].head if gold[word].head > word else s.n
    for buff_i in range(s.i, stop):
        if gold[buff_i].head == word:
            return 1
    return 0


cdef int has_head_in_buffer(State *s, size_t word, Token* gold) except -1:
    assert word != 0
    cdef size_t buff_i
    for buff_i in range(s.i, s.n):
        if gold[word].head == buff_i:
            return 1
    return 0


cdef int has_child_in_stack(State *s, size_t word, Token* gold) except -1:
    assert word != 0
    cdef size_t i, stack_i
    cdef int n = 0
    for i in range(s.stack_len):
        stack_i = s.stack[i]
        # Should this be sensitive to whether the word has a head already?
        if gold[stack_i].head == word:
            n += 1
    return n


cdef int has_head_in_stack(State *s, size_t word, Token* gold) except -1:
    assert word != 0
    cdef size_t i, stack_i
    for i in range(s.stack_len):
        stack_i = s.stack[i]
        if gold[word].head == stack_i:
            return 1
    return 0


cdef bint at_eol(State *s):
    return s.i >= (s.n - 1)

cdef bint is_final(State *s):
    return at_eol(s) and s.stack_len == 0

DEF PADDING = 5


cdef State* init_state(size_t length) except NULL:
    cdef size_t i
    cdef State* s = <State*>calloc(1, sizeof(State))
    s.n = length
    s.m = 0
    s.i = 0
    s.cost = 0
    s.score = 0
    s.string_prob = 0 
    s.top = 0
    s.stack_len = 0
    n = length + PADDING
    s.stack = <size_t*>calloc(n, sizeof(size_t))
    s.l_children = <size_t**>malloc(n * sizeof(size_t*))
    s.r_children = <size_t**>malloc(n * sizeof(size_t*))
    s.parse = <Token*>calloc(n, sizeof(Token))
    for i in range(n):
        s.parse[i].i = i
        s.parse[i].left_edge = i
        s.parse[i].word = &BLANK_WORD
        s.l_children[i] = <size_t*>calloc(MAX_VALENCY, sizeof(size_t))
        s.r_children[i] = <size_t*>calloc(MAX_VALENCY, sizeof(size_t))
    s.history = <Transition*>calloc(n * 20, sizeof(Transition))
    return s


cdef int copy_state(State* s, State* old) except -1:
    cdef size_t nbytes, i
    if s.i > old.i:
        nbytes = (s.i + 1) * sizeof(size_t)
    else:
        nbytes = (old.i + 1) * sizeof(size_t)
    s.score = old.score
    s.string_prob = old.string_prob
    s.i = old.i
    s.m = old.m
    s.n = old.n
    s.stack_len = old.stack_len
    s.top = old.top
    s.cost = old.cost
    # Be thrifty in what we copy, as in large beams it starts to matter 
    memcpy(s.stack, old.stack, (old.stack_len + 1) * sizeof(size_t))
    # Only have to look up to (and including) the start of the buffer 
    for i in range(old.i + 1):
        memcpy(s.l_children[i], old.l_children[i], old.parse[i].l_valency * sizeof(size_t))
        memcpy(s.r_children[i], old.r_children[i], old.parse[i].r_valency * sizeof(size_t))
    # TODO: This seems to change feature calculations, if we limit to (old.i + 1)
    # Why?
    memcpy(s.parse, old.parse, old.n * sizeof(Token))
    memcpy(s.history, old.history, old.m * sizeof(Transition))


cdef free_state(State* s):
    free(s.stack)
    for i in range(s.n + PADDING):
        free(s.l_children[i])
        free(s.r_children[i])
    free(s.history)
    free(s.l_children)
    free(s.r_children)
    free(s.parse)
    free(s)
