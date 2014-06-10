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


cdef bint can_shift(State* s):
    return not at_eol(s)


cdef bint can_right(State* s):
    return s.stack_len >= 2


cdef bint can_left(State* s):
    return s.stack_len >= 1


cdef bint USE_EDIT = False
cdef bint can_edit(State* s):
    return s.stack_len


cdef bint USE_BREAK = False
cdef bint can_break(State* s):
    return USE_BREAK and s.stack_len == 1 and not s.parse[s.i].l_valency and not at_eol(s)


# Edit oracle:
# - You can always LeftArc from an Edit word
# - You can always RightArc between two Edit words
# - You can never arc from a fluent word to a disfluent word
# - You can't RightArc from a disfluent word to a fluent word
# - You can always Shift an Edit word


cdef int shift_cost(State* s, Token* gold) except -1:
    assert not at_eol(s)
    cost = 0
    cdef size_t w
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].head == s.top:
        return cost
    if gold[s.i].is_edit:
        return cost
    cost += has_head_in_stack(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    return cost

cdef int shift_label_cost(State* s, size_t label, Step* lattice, Token* gold) except -1:
    # We'll shift the word at s.i+1 in from the lattice, choosing the word indicated
    # by label. So, check that either s.i+1 is an edit, or we're shifting in the
    # right word
    if (s.i + 1) < s.n and not gold[s.i+1].is_edit and \
      gold[s.i+1].word != lattice[s.i+1].nodes[label]:
        return 1
    else:
        return 0
 

cdef int right_cost(State* s, Token* gold) except -1:
    cost = 0
    cdef size_t s1 = get_s1(s)
    if gold[s1].is_edit and gold[s.top].is_edit:
        return cost
    elif gold[s1].is_edit or gold[s.top].is_edit:
        cost += 1
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int right_label_cost(State* s, size_t label, Step* lattice, Token* gold) except -1:
    return gold[s.top].head == get_s1(s) and gold[s.top].label != label


cdef int left_cost(State* s, Token* gold) except -1:
    assert s.stack_len
    cost = 0
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].is_edit:
        return cost
    if not gold[s.i].is_edit and gold[s.top].is_edit:
        return cost + 1
    if gold[s.top].head == s.i:
        return cost
    cost += gold[s.top].head == get_s1(s)
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int left_label_cost(State* s, size_t label, Step* lattice, Token* gold) except -1:
    return gold[s.top].head == s.i and gold[s.top].label != label


cdef int edit_cost(State *s, Token* gold) except -1:
    return not gold[s.top].is_edit

cdef int edit_label_cost(State* s, size_t label, Step* lattice, Token* gold) except -1:
    return 0


cdef int break_cost(State* s, Token* gold) except -1:
    assert s.stack_len == 1
    assert not at_eol(s)
    return 0 if gold[s.top].sent_id != gold[s.i].sent_id else 1


cdef int break_label_cost(State* s, size_t label, Step* lattice, Token* gold) except -1:
    return 0



cdef int fill_valid(State* s, Step* lattice, Transition* classes, size_t n) except -1:
    cdef Transition* t
    cdef size_t lattice_n = lattice[s.i+1].n
    cdef bint[N_MOVES] validity
    validity[SHIFT] = can_shift(s)
    validity[RIGHT] = can_right(s)
    validity[LEFT] = can_left(s)
    validity[EDIT] = can_edit(s)
    validity[BREAK] = can_break(s)
    seen_valid = True
    for i in range(n):
        t = &classes[i]
        t.is_valid = validity[t.move]
        if t.move == SHIFT and t.is_valid:
            t.is_valid = t.label < lattice_n 
        if t.is_valid:
            seen_valid = True
    if not seen_valid:
        print s.i, s.n, s.stack_len, is_final(s)
        raise StandardError

ctypedef int (*label_cost_func)(State* s, size_t label, Step* lattice, Token* gold) except -1

cdef int fill_costs(State* s, Step* lattice, Transition* classes,
                    size_t n, Token* gold) except -1:
    cdef bint[N_MOVES] validity
    validity[SHIFT] = can_shift(s)
    validity[RIGHT] = can_right(s)
    validity[LEFT] = can_left(s)
    validity[EDIT] = can_edit(s)
    validity[BREAK] = can_break(s)
 
    cdef int costs[N_MOVES]
    costs[SHIFT] = shift_cost(s, gold) if validity[SHIFT] else 0
    costs[LEFT] = left_cost(s, gold) if validity[LEFT] else 0
    costs[RIGHT] = right_cost(s, gold) if validity[RIGHT] else 0
    costs[EDIT] = edit_cost(s, gold) if validity[EDIT] else 0
    costs[BREAK] = break_cost(s, gold) if validity[BREAK] else 0

    cdef label_cost_func[N_MOVES] label_costs
    # These get their own functions for efficiency, so that we don't have to call
    # the cost functions for each labelled move --- we memoise the move costs,
    # and invoke the label costs for zero-cost moves.
    label_costs[SHIFT] = shift_label_cost
    label_costs[RIGHT] = right_label_cost
    label_costs[LEFT] = left_label_cost
    label_costs[EDIT] = edit_label_cost
    label_costs[BREAK] = break_label_cost

    cdef size_t lattice_n = lattice[s.i+1].n
    cdef size_t i
    cdef Transition* t
    for i in range(n):
        t = &classes[i]
        t.cost = costs[t.move]
        if t.move == SHIFT and t.label >= lattice_n:
            t.is_valid = False
            t.cost = 0
        elif t.cost == 0:
            t.cost += label_costs[t.move](s, t.label, lattice, gold)


cdef int transition(Transition* t, State *s, Step* lattice) except -1:
    s.history[s.m] = t[0]
    s.m += 1 
    s.string_prob = 0
    if t.move == SHIFT:
        push_stack(s, t.label, lattice)
        if lattice[s.i].probs[t.label] >= 0:
            s.string_prob = 1
        else:
            s.string_prob = lattice[s.i].probs[t.label]
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


cdef size_t get_nr_moves(size_t shift_classes, size_t lattice_width,
                         list left_labels, list right_labels,
                         list dfl_labels, bint use_break):
    global USE_BREAK, USE_EDIT
    USE_BREAK = use_break
    USE_EDIT = bool(dfl_labels) 
    return lattice_width + use_break + len(left_labels) + len(right_labels) + len(dfl_labels)


cdef int fill_moves(size_t shift_classes, size_t lattice_width, list left_labels,
                    list right_labels, list dfl_labels, bint use_break, Transition* moves):
    cdef size_t label
    cdef size_t i = 0
    cdef size_t clas = 0
    cdef size_t root_label = index.hashes.encode_label(b'ROOT')
    # These Shift moves are distinct as far as the learner is concerned;
    # they receive their own feature space.
    for label in range(shift_classes):
        moves[i].move = SHIFT; moves[i].label = label; i += 1
        moves[i].clas = clas; clas += 1
    # These Shift moves are collapsed together as far as the learner is
    # concerned, we search between them with the beam.
    for label in range(shift_classes, lattice_width):
        moves[i].move = SHIFT; moves[i].label = label; i += 1
        moves[i].clas = clas # DON'T increment class for lattice moves
    clas += 1
    if use_break:
        moves[i].move = BREAK; moves[i].label = root_label; i += 1
        moves[i].clas = clas; clas += 1
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
