# cython: profile=True
from _state cimport *
from libc.stdlib cimport malloc, calloc, free
import index.hashes

# TODO: Link these with other compile constants
DEF MAX_TAGS = 100
DEF MAX_LABELS = 200


cdef enum:
    ERR
    SHIFT
    LEFT
    RIGHT
    EDIT
    FILLER
    BREAK
    N_MOVES


cdef inline bint can_shift(State* s):
    return not s.at_end_of_buffer


cdef inline bint can_right(State* s):
    return s.stack_len >= 2


cdef inline bint can_left(State* s):
    return s.stack_len >= 1


cdef bint USE_EDIT = False
cdef inline bint can_edit(State* s):
    return USE_EDIT and s.stack_len


cdef bint USE_BREAK = False
cdef inline bint can_break(State* s):
    return USE_BREAK and s.stack_len == 1 and not s.parse[s.i].l_valency and not s.at_end_of_buffer


cdef bint USE_FILL = False
cdef inline bint can_filler(State* s):
    return s.stack_len >= 1


# Edit oracle:
# - You can always LeftArc from an Edit word
# - You can always RightArc between two Edit words
# - You can never arc from a fluent word to a disfluent word
# - You can't RightArc from a disfluent word to a fluent word
# - You can always Shift an Edit word

cdef int shift_cost(State* s, Token* gold):
    assert not s.at_end_of_buffer
    cost = 0
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].head == s.top:
        return cost
    if gold[s.i].is_edit:
        return cost
    cost += has_head_in_stack(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    return cost


cdef int right_cost(State* s, Token* gold):
    assert s.stack_len >= 2
    assert not can_break(s)
    cost = 0
    cost += gold[s.top].is_fill
    if gold[s.second].is_edit and gold[s.top].is_edit:
        return 1
    elif gold[s.second].is_edit or gold[s.top].is_edit:
        cost += 1
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int left_cost(State* s, Token* gold):
    assert s.stack_len
    cost = 0
    cost += gold[s.top].is_fill
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].is_edit:
        return cost
    if not gold[s.i].is_edit and gold[s.top].is_edit:
        return cost + 1
    if gold[s.top].head == s.i:
        return cost
    cost += gold[s.top].head == s.second
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int edit_cost(State *s, Token* gold):
    assert s.stack_len >= 1
    return 0 if gold[s.top].is_edit else 1


cdef int break_cost(State* s, Token* gold):
    assert s.stack_len == 1
    assert not s.at_end_of_buffer
    return 0 if gold[s.top].sent_id != gold[s.i].sent_id else 1


cdef int filler_cost(State* s, Token* gold):
    assert s.stack_len >= 1
    return 0 if gold[s.top].is_fill else 1


cdef int fill_valid(State* s, Transition* classes, size_t n) except -1:
    cdef bint[N_MOVES] valid
    valid[SHIFT] = can_shift(s)
    valid[LEFT] = can_left(s)
    valid[RIGHT] = can_right(s)
    valid[EDIT] = can_edit(s)
    valid[BREAK] = can_break(s)
    valid[FILLER] = can_filler(s)
    for i in range(n):
        classes[i].is_valid = valid[classes[i].move]
    for i in range(n):
        if classes[i].is_valid:
            break
    else:
        raise StandardError


cdef int fill_costs(State* s, Transition* classes, size_t n, Token* gold) except -1:
    cdef int[N_MOVES] costs
    costs[SHIFT] = shift_cost(s, gold) if can_shift(s) else -1
    costs[LEFT] = left_cost(s, gold) if can_left(s) else -1
    costs[RIGHT] = right_cost(s, gold) if can_right(s) else -1
    costs[EDIT] = edit_cost(s, gold) if can_edit(s) else -1
    costs[BREAK] = break_cost(s, gold) if can_break(s) else -1
    costs[FILLER] = filler_cost(s, gold) if can_filler(s) else -1
    #print costs[SHIFT], costs[LEFT], costs[RIGHT], costs[EDIT]
    for i in range(n):
        classes[i].cost = costs[classes[i].move]
        if classes[i].move == LEFT and classes[i].cost == 0:
            classes[i].cost += gold[s.top].label != classes[i].label
        elif classes[i].move == RIGHT and classes[i].cost == 0:
            classes[i].cost = gold[s.top].label != classes[i].label


cdef int transition(Transition* t, State *s) except -1:
    assert not s.is_finished
    s.history[s.m] = t[0]
    s.m += 1 
    if t.move == SHIFT:
        push_stack(s)
    elif t.move == LEFT:
        add_dep(s, s.i, s.top, t.label)
        pop_stack(s)
    elif t.move == RIGHT:
        add_dep(s, s.second, s.top, t.label)
        pop_stack(s)
    elif t.move == EDIT:
        edited = pop_stack(s)
        while s.parse[edited].l_valency:
            child = get_l(s, edited)
            del_l_child(s, edited)
            s.second = s.top
            s.top = child
            s.stack[s.stack_len] = child
            s.stack_len += 1
        for i in range(edited, s.parse[s.i].left_edge):
            if not s.parse[i].is_fill:
                s.parse[i].head = i
                s.parse[i].label = t.label
                s.parse[i].is_edit = True
    elif t.move == BREAK:
        assert s.stack_len == 1
        add_dep(s, s.n - 1, s.top, t.label)
        s.parse[s.i].sent_id = s.parse[s.top].sent_id + 1
        pop_stack(s)
    elif t.move == FILLER:
        assert s.stack_len >= 1
        add_dep(s, s.n - 1, s.top, t.label)
        s.parse[s.top].is_fill = True
        pop_stack(s)
    else:
        raise StandardError(t.move)
    if s.i >= (s.n - 1):
        s.at_end_of_buffer = True
    if s.at_end_of_buffer and s.stack_len == 0:
        s.is_finished = True


cdef size_t get_nr_moves(list left_labels, list right_labels,
                         bint use_edit, bint use_break, bint use_fill):
    global USE_BREAK, USE_EDIT
    USE_BREAK = use_break
    USE_EDIT = use_edit
    USE_FILL = use_fill
    return 1 + use_edit + use_break + use_fill + len(left_labels) + len(right_labels)


cdef int fill_moves(list left_labels, list right_labels, bint use_edit,
                    bint use_break, bint use_fill, Transition* moves):
    cdef size_t i = 0
    cdef size_t erase_label = index.hashes.encode_label('erased')
    cdef size_t root_label = index.hashes.encode_label('ROOT')
    cdef size_t filler_label = index.hashes.encode_label('filler')
    moves[i].move = SHIFT; moves[i].label = 0; i += 1
    if use_edit:
        moves[i].move = EDIT; moves[i].label = erase_label; i += 1
    if use_fill:
        moves[i].move = FILLER; moves[i].label = filler_label; i += 1
    if use_break:
        moves[i].move = BREAK; moves[i].label = root_label; i += 1
    cdef size_t label
    for label in left_labels:
        moves[i].move = LEFT; moves[i].label = label; i += 1
    for label in right_labels:
        moves[i].move = RIGHT; moves[i].label = label; i += 1
    cdef size_t clas
    for clas in range(i):
        moves[clas].clas = clas
        moves[clas].score = 0
        moves[clas].cost = 0
        moves[clas].is_valid = True
