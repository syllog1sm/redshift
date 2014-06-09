# cython: profile=True
from _state cimport *
from libc.stdlib cimport malloc, calloc, free
import index.hashes
from redshift.sentence cimport Step

from libcpp.vector cimport vector

# TODO: Link these with other compile constants
DEF MAX_TAGS = 100
DEF MAX_LABELS = 200


cdef enum:
    ERR
    SHIFT
    LEFT
    RIGHT
    EDIT
    BREAK
    N_MOVES


cdef bint can_shift(State* s, Step* lattice, size_t label):
    return not at_eol(s) and label < lattice[s.i + 1].n


cdef bint can_right(State* s, Step* lattice, size_t label):
    return s.stack_len >= 2


cdef bint can_left(State* s, Step* lattice, size_t label):
    return s.stack_len >= 1


cdef bint USE_EDIT = False
cdef bint can_edit(State* s, Step* lattice, size_t label):
    return s.stack_len


cdef bint USE_BREAK = False
cdef bint can_break(State* s, Step* lattice, size_t label):
    return USE_BREAK and s.stack_len == 1 and not s.parse[s.i].l_valency and not at_eol(s)


# Edit oracle:
# - You can always LeftArc from an Edit word
# - You can always RightArc between two Edit words
# - You can never arc from a fluent word to a disfluent word
# - You can't RightArc from a disfluent word to a fluent word
# - You can always Shift an Edit word


cdef int shift_cost(State* s, size_t label, Token* gold, Step* lattice) except -1:
    assert not at_eol(s)
    cost = 0
    cdef size_t w
    if can_break(s, lattice, 0):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    # We'll shift the word at s.i+1 in from the lattice, choosing the word indicated
    # by label. So, check that either s.i+1 is an edit, or we're shifting in the
    # right word
    if (s.i + 1) < s.n and not gold[s.i+1].is_edit and \
      gold[s.i+1].word != lattice[s.i+1].nodes[label]:
        cost += 1
    if gold[s.i].head == s.top:
        return cost
    if gold[s.i].is_edit:
        return cost
    cost += has_head_in_stack(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    return cost


cdef int right_cost(State* s, size_t label, Token* gold, Step* lattice) except -1:
    assert s.stack_len >= 2
    assert not can_break(s, lattice, 0)
    cost = 0
    if gold[get_s1(s)].is_edit and gold[s.top].is_edit:
        return cost
    elif gold[get_s1(s)].is_edit or gold[s.top].is_edit:
        cost += 1
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    if gold[s.top].head == get_s1(s) and gold[s.top].label != label:
        cost += 1
    return cost


cdef int left_cost(State* s, size_t label, Token* gold, Step* lattice) except -1:
    assert s.stack_len
    cost = 0
    if can_break(s, lattice, 0):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].is_edit:
        return cost
    if not gold[s.i].is_edit and gold[s.top].is_edit:
        return cost + 1
    if gold[s.top].head == s.i:
        cost += gold[s.top].label != label
        return cost
    cost += gold[s.top].head == get_s1(s)
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int edit_cost(State *s, size_t label, Token* gold, Step* lattice) except -1:
    # TODO: Why is being label agnostic here a problem?
    if gold[s.top].is_edit:
        return 0
    else:
        return 1


cdef int break_cost(State* s, size_t label, Token* gold, Step* lattice) except -1:
    assert s.stack_len == 1
    assert not at_eol(s)
    return 0 if gold[s.top].sent_id != gold[s.i].sent_id else 1


ctypedef int (*cost_func)(State* s, size_t label, Token* gold, Step* lattice) except -1

cdef cost_func[N_MOVES] cost_getters

cost_getters[SHIFT] = shift_cost
cost_getters[RIGHT] = right_cost
cost_getters[LEFT] = left_cost
cost_getters[EDIT] = edit_cost
cost_getters[BREAK] = break_cost

ctypedef bint (*can_func)(State* s, Step* lattice, size_t label)

cdef can_func[N_MOVES] valid_checkers

valid_checkers[SHIFT] = can_shift
valid_checkers[RIGHT] = can_right
valid_checkers[LEFT] = can_left
valid_checkers[EDIT] = can_edit
valid_checkers[BREAK] = can_break

cdef int fill_valid(State* s, Step* lattice, Transition* classes, size_t n) except -1:
    cdef Transition* t
    for i in range(n):
        t = &classes[i]
        t.is_valid = valid_checkers[t.move](s, lattice, t.label)
    for i in range(n):
        if classes[i].is_valid:
            break
    else:
        print s.i, s.n, s.stack_len, is_final(s)
        raise StandardError


cdef int fill_costs(State* s, Step* lattice, Transition* classes,
                    size_t n, Token* gold) except -1:
    cdef Transition* t
    for i in range(n):
        t = &classes[i]
        if valid_checkers[t.move](s, lattice, t.label):
            t.cost = cost_getters[t.move](s, t.label, gold, lattice)
        else:
            t.cost = -1


cdef int transition(Transition* t, State *s, Step* lattice) except -1:
    s.history[s.m] = t[0]
    s.m += 1 
    if t.move == SHIFT:
        push_stack(s, t.label, lattice)
    elif t.move == LEFT:
        add_dep(s, s.i, s.top, t.label)
        pop_stack(s)
    elif t.move == RIGHT:
        add_dep(s, get_s1(s), s.top, t.label)
        pop_stack(s)
    elif t.move == EDIT:
        edited = pop_stack(s)
        while s.parse[edited].l_valency:
            child = get_l(s, edited)
            del_l_child(s, edited)
            s.top = child
            s.stack[s.stack_len] = child
            s.stack_len += 1
        for i in range(edited, s.parse[s.i].left_edge):
            # We might have already set these as edits, under a different
            # label.
            if s.parse[i].is_edit:
                break
            s.parse[i].head = i
            s.parse[i].label = t.label
            s.parse[i].is_edit = True
    elif t.move == BREAK:
        assert s.stack_len == 1
        add_dep(s, s.n - 1, s.top, t.label)
        s.parse[s.i].sent_id = s.parse[s.top].sent_id + 1
        pop_stack(s)
    else:
        raise StandardError(t.move)


cdef size_t get_nr_moves(size_t lattice_width, list left_labels, list right_labels,
                         list dfl_labels, bint use_break):
    global USE_BREAK, USE_EDIT
    USE_BREAK = use_break
    USE_EDIT = bool(dfl_labels) 
    return lattice_width + use_break + len(left_labels) + len(right_labels) + len(dfl_labels)


cdef int fill_moves(size_t lattice_width, list left_labels, list right_labels,
                    list dfl_labels, bint use_break, Transition* moves):
    cdef size_t root_label = index.hashes.encode_label('ROOT')
    cdef size_t i = 0
    cdef size_t clas = 0
    for i in range(lattice_width):
        moves[i].move = SHIFT
        moves[i].label = i
        moves[i].clas = clas
    i += 1; clas += 1
    if use_break:
        moves[i].move = BREAK; moves[i].label = root_label; i += 1
        moves[i].clas = clas; clas += 1
    cdef size_t label
    for label in dfl_labels:
        moves[i].move = EDIT; moves[i].label = label; i += 1
        moves[i].clas = clas; clas += 1
    for label in left_labels:
        moves[i].move = LEFT; moves[i].label = label; i += 1
        moves[i].clas = clas; clas += 1
    for label in right_labels:
        moves[i].move = RIGHT; moves[i].label = label; i += 1
        moves[i].clas = clas; clas += 1
    for m in range(i):
        moves[m].score = 0
        moves[m].cost = 0
        moves[m].is_valid = True
