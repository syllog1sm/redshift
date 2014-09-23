# cython: profile=True
from _state cimport *
import index.hashes

# TODO: Link these with other compile constants
DEF MAX_TAGS = 100
DEF MAX_LABELS = 200


cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    N_MOVES


cdef unicode move_name(Transition* t):
    moves = [u'E', u'S', u'D', u'L', u'R']
    name = moves[t.move]
    if t.move == RIGHT or t.move == LEFT:
        name += u'-%s' % index.hashes.decode_label(t.label)
    return name


cdef inline bint can_shift(State* s):
    return not at_eol(s)


cdef inline bint can_right(State* s):
    return s.stack_len >= 1 and not at_eol(s)


cdef inline bint can_left(State* s):
    return s.stack_len >= 1 and s.parse[s.top].head == 0


cdef inline bint can_reduce(State* s):
    return s.stack_len >= 2 and s.parse[s.top].head != 0

cdef int shift_cost(State* s, Token* gold):
    assert not at_eol(s)
    cost = 0
    cost += has_head_in_stack(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    return cost


cdef int right_cost(State* s, Token* gold):
    assert s.stack_len >= 1
    if gold[s.i].head == s.top:
        return 0
    cost = 0
    cost += has_head_in_buffer(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    cost += has_head_in_stack(s, s.i, gold)
    return cost


cdef int left_cost(State* s, Token* gold):
    assert s.stack_len >= 1
    cost = 0
    if gold[s.top].head == s.i:
        return cost
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int reduce_cost(State* s, Token* gold):
    assert s.stack_len >= 2
    cost = 0
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int fill_valid(State* s, Transition* classes, size_t n) except -1:
    cdef bint[N_MOVES] valid
    valid[SHIFT] = can_shift(s)
    valid[LEFT] = can_left(s)
    valid[RIGHT] = can_right(s)
    valid[REDUCE] = can_reduce(s)
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
    costs[REDUCE] = reduce_cost(s, gold) if can_reduce(s) else -1
    for i in range(n):
        classes[i].cost = costs[classes[i].move]
        if classes[i].move == LEFT and classes[i].cost == 0 and \
          gold[s.top].head == s.i:
            classes[i].cost += gold[s.top].label != classes[i].label
        elif classes[i].move == RIGHT and classes[i].cost == 0 and \
          gold[s.i].head == s.top:
            classes[i].cost += gold[s.i].label != classes[i].label


cdef int transition(Transition* t, State *s) except -1:
    s.history[s.m] = t[0]
    s.m += 1 
    if t.move == SHIFT:
        push_stack(s)
    elif t.move == LEFT:
        add_dep(s, s.i, s.top, t.label)
        pop_stack(s)
    elif t.move == RIGHT:
        add_dep(s, s.top, s.i, t.label)
        push_stack(s)
    elif t.move == REDUCE:
        pop_stack(s)
    else:
        raise StandardError(t.move)


cdef size_t get_nr_moves(list left_labels, list right_labels, list dfl_labels,
                         bint use_break):
    assert not dfl_labels
    assert not use_break
    return 2 + len(left_labels) + len(right_labels)


cdef int fill_moves(list left_labels, list right_labels, list dfl_labels,
        bint use_break, Transition* moves) except -1:
    assert not dfl_labels
    assert not use_break
    cdef size_t i = 0
    cdef size_t root_label = index.hashes.encode_label('ROOT')
    moves[i].move = SHIFT; moves[i].label = 0; i += 1
    moves[i].move = REDUCE; moves[i].label = 0; i += 1
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
