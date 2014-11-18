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
    EDIT
    N_MOVES


cdef unicode move_name(Transition* t):
    moves = [u'?', u'S', u'D', u'L', u'R', u'E']
    name = moves[t.move]
    if t.move == RIGHT or t.move == LEFT or t.move == EDIT:
        name += u'-%s' % index.hashes.decode_label(t.label)
    return name


cdef inline bint can_shift(State* s) nogil:
    return not at_eol(s)


cdef inline bint can_right(State* s) nogil:
    return s.stack_len >= 1 and not at_eol(s)


cdef inline bint can_left(State* s) nogil:
    return s.stack_len >= 1 and s.parse[s.top].head == 0


cdef inline bint can_reduce(State* s) nogil:
    return s.stack_len >= 2 and s.parse[s.top].head != 0


cdef bint USE_EDIT = False
cdef inline bint can_edit(State* s) nogil:
    return USE_EDIT and s.stack_len >= 1


# Edit oracle:
# - You can always Shift a disfluent word
# - You can never Reduce a disfluent word headed by a fluent word
# - You can always LeftArc from a disfluent word to a fluent word
# - You can always RightArc to a disfluent word from a fluent word
# - You can never LeftArc from a fluent word to a disfluent word
# - You can never RightArc from a disfluent word to a fluent word
# - You can always RightArc from a disfluent word to a disfluent word
# - You can only LeftArc from a disfluent word to a disfluent word if it has no
#   fluent children
# - You can only Reduce a disfluent word if its head is disfluent
cdef int shift_cost(State* s, Token* gold):
    assert not at_eol(s)
    cost = 0
    # - You can always Shift a disfluent word
    if gold[s.i].is_edit:
        return cost
    cost += has_head_in_stack(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    return cost


cdef int right_cost(State* s, Token* gold):
    assert s.stack_len >= 1
    cost = 0
    if gold[s.i].head == s.top:
        return cost
    # - You can always RightArc to a disfluent word
    if gold[s.i].is_edit:
        return cost
    # - You can't RightArc from a disfluent word to a fluent word
    elif gold[s.top].is_edit or gold[s.i].is_edit:
        cost += 1
    cost += has_head_in_buffer(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    cost += has_head_in_stack(s, s.i, gold)
    return cost


cdef int left_cost(State* s, Token* gold):
    assert s.stack_len >= 1
    cost = 0
    # - You can always LeftArc from a disfluent word to a fluent word
    if gold[s.i].is_edit and not gold[s.top].is_edit:
        return cost
    # - You can only LeftArc from a disfluent word to a disfluent word if it has no
    #   fluent children
    if gold[s.i].is_edit and gold[s.top].is_edit:
        for i in range(s.parse[s.top].l_valency):
            if not gold[s.l_children[s.top][i]].is_edit:
                cost += 1
    # - You can never arc from a fluent word to a disfluent word
    if not gold[s.i].is_edit and gold[s.top].is_edit:
        cost += 1
    if gold[s.top].head == s.i:
        return cost
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int reduce_cost(State* s, Token* gold):
    assert s.stack_len >= 2
    cost = 0
    if gold[s.top].is_edit and not gold[s.parse[s.top].head].is_edit:
        cost += 1
    cost += has_child_in_buffer(s, s.top, gold)
    return cost


cdef int edit_cost(State* s, Token* gold):
    assert s.stack_len >= 1
    return 0 if gold[s.top].is_edit else 1


cdef int fill_valid(State* s, Transition* classes, size_t n) except -1:
    cdef bint[N_MOVES] valid
    valid[SHIFT] = can_shift(s)
    valid[LEFT] = can_left(s)
    valid[RIGHT] = can_right(s)
    valid[REDUCE] = can_reduce(s)
    valid[EDIT] = can_edit(s)
    for i in range(n):
        classes[i].is_valid = valid[classes[i].move]
    for i in range(n):
        if classes[i].is_valid:
            break
    else:
        props = (s.i, s.n, s.stack_len)
        raise StandardError('No valid classes. i=%d, n=%d, stack_len=%d' % props)


cdef int fill_costs(State* s, Transition* classes, size_t n, Token* gold) except -1:
    cdef int[N_MOVES] costs
    costs[SHIFT] = shift_cost(s, gold) if can_shift(s) else -1
    costs[LEFT] = left_cost(s, gold) if can_left(s) else -1
    costs[RIGHT] = right_cost(s, gold) if can_right(s) else -1
    costs[REDUCE] = reduce_cost(s, gold) if can_reduce(s) else -1
    costs[EDIT] = edit_cost(s, gold) if can_edit(s) else -1
    for i in range(n):
        classes[i].cost = costs[classes[i].move]
        if classes[i].move == LEFT and classes[i].cost == 0 and \
          gold[s.top].head == s.i:
            classes[i].cost += gold[s.top].label != classes[i].label
        elif classes[i].move == RIGHT and classes[i].cost == 0 and \
          gold[s.i].head == s.top:
            classes[i].cost += gold[s.i].label != classes[i].label
        elif classes[i].move == EDIT and classes[i].cost == 0 and gold[s.top].is_edit:
            classes[i].cost = gold[s.top].label != classes[i].label
        # Set is_valid here as well, so we can just over-write it if we don't
        # want to follow gold
        classes[i].is_valid = classes[i].cost == 0


cdef int transition(Transition* t, State *s) except -1:
    cdef size_t edited
    cdef size_t child
    cdef size_t i
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
    else:
        raise StandardError(t.move)


cdef size_t get_nr_moves(list left_labels, list right_labels, list dfl_labels,
                         bint use_break):
    global USE_EDIT
    USE_EDIT = bool(dfl_labels) 
    return 2 + len(left_labels) + len(right_labels) + len(dfl_labels)


cdef int fill_moves(list left_labels, list right_labels, list dfl_labels,
        bint use_break, Transition* moves) except -1:
    assert not use_break
    cdef size_t i = 0
    cdef size_t root_label = index.hashes.encode_label('ROOT')
    moves[i].move = SHIFT; moves[i].label = 0; i += 1
    moves[i].move = REDUCE; moves[i].label = 0; i += 1
    cdef size_t label
    for label in dfl_labels:
        moves[i].move = EDIT; moves[i].label = label; i += 1
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
