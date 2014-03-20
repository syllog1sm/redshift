# cython: profile=True
from _state cimport *
from libc.stdlib cimport malloc, calloc, free

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
    BREAK
    N_MOVES


cdef inline bint can_shift(State* s):
    return not s.at_end_of_buffer and not s.segment


cdef inline bint can_right(State* s):
    return s.stack_len and not s.at_end_of_buffer and not s.segment


cdef inline bint can_reduce(State* s):
    return s.stack_len and s.parse[s.top].head


cdef inline bint can_left(State* s):
    return s.stack_len and not s.segment and s.parse[s.top].head == 0


cdef inline bint can_edit(State* s):
    return s.stack_len and not s.segment


cdef inline bint can_break(State* s):
    #if not sbd:
    #    return False
    if s.at_end_of_buffer:
        return False
    elif s.segment:
        return False
    elif not s.stack_len:
        return False
    elif not s.m:
        return False
    #elif moves[s.history[s.t-1]] != SHIFT and moves[s.history[s.t-1]] != RIGHT:
    #    return False
    elif nr_headless(s) != 1:
        return False
    else:
        return True


cdef int shift_cost(State *s, AnswerToken* gold):
    cdef int cost = 0
    if s.stack_len < 1:
        return 0
    # Be flexible about sentence boundaries around disfluencies
    if gold[s.i].is_edit:
        return cost
    if can_break(s):
        cost += gold[s.top].is_break
    cost += has_child_in_stack(s, s.i, gold)
    cost += has_head_in_stack(s, s.i, gold)
    return cost


cdef int right_cost(State *s, AnswerToken* gold):
    cdef int cost = 0
    if gold[s.top].is_edit and not gold[s.i].is_edit:
        return 1
    if gold[s.i].is_edit:
        return cost
    if can_break(s):
        cost += gold[s.top].is_break
    if gold[s.i].head == s.top:
        return cost
    # TODO: ???
    # This probably allows flexibility for attachment direction for segmentation
    #if heads[s.i] == s.heads[s.top] == self.root_label:
    #    return cost
    cost += has_head_in_buffer(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    cost += has_head_in_stack(s, s.i, gold)
    return cost


cdef int reduce_cost(State *s, AnswerToken* gold):
    cdef int cost = 0
    if s.segment:
        return 0
    if can_break(s):
        cost += gold[s.top].is_break
    cost += has_child_in_buffer(s, s.top, gold)
    #if self.allow_reattach:
    #    cost += has_head_in_buffer(s, s.top, gold)
    if gold[s.top].is_edit and not gold[s.parse[s.top].head].is_edit:
        cost += 1
    return cost


cdef int left_cost(State *s, AnswerToken* gold) except -9000:
    cdef int cost = 0
    if can_break(s):
        cost += gold[s.top].is_break
    # This would form a dep between an edit and non-edit word
    if gold[s.top].is_edit and not gold[s.i].is_edit:
         cost += 1
    elif gold[s.i].is_edit:
        return cost
    if gold[s.top].head == s.i:
        return cost
    # TODO: This is supposed to give dynamic oracle flexibility when we're
    # not using the B transition. But, the line as written doesn't make sense.
    #elif not self.use_sbd and heads[s.top] == heads[s.i] == self.root_label:
    #    return cost
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    #if self.allow_reattach and gold[s.top].head == s.parse[s.top].head:
    #    cost += 1
    #if self.allow_reduce and gold[s.top].head == s.second:
    #    cost += 1
    return cost


cdef int break_cost(State *s, AnswerToken* gold):
    return not gold[s.top].is_break


cdef int edit_cost(State *s, AnswerToken* gold):
    cdef int cost = 0
    #if can_segment(s, self.moves, self.use_sbd):
    #    cost += sbd[s.top] != sbd[s.i]
    cost += not gold[s.top].is_edit
    return cost


cdef int fill_valid(State* s, Transition* classes, size_t n) except -1:
    cdef bint[N_MOVES] valid
    valid[SHIFT] = can_shift(s)
    valid[REDUCE] = can_reduce(s)
    valid[LEFT] = can_left(s)
    valid[RIGHT] = can_right(s)
    valid[EDIT] = can_edit(s)
    #valid[BREAK] = can_break(s)
    for i in range(n):
        classes[i].is_valid = valid[classes[i].move]


cdef int fill_costs(State* s, Transition* classes, size_t n, AnswerToken* gold) except -1:
    cdef int[N_MOVES] costs
    costs[SHIFT] = can_shift(s) and shift_cost(s, gold) == 0
    costs[REDUCE] = can_reduce(s) and reduce_cost(s, gold) == 0
    costs[LEFT] = can_left(s) and left_cost(s, gold) == 0
    costs[RIGHT] = can_right(s) and right_cost(s, gold) == 0
    costs[EDIT] = can_edit(s) and edit_cost(s, gold) == 0
    #costs[BREAK] = can_break(s) and break_cost(s, gold) == 0
    for i in range(n):
        classes[i].cost = costs[classes[i].move]


cdef int transition(Transition* t, State *s) except -1:
    if s.stack_len >= 1:
        assert s.top != 0
    assert not s.is_finished
    s.history[s.m] = t[0]
    s.m += 1 
    if t.move == SHIFT:
        push_stack(s)
    elif t.move == REDUCE:
        pop_stack(s)
    elif t.move == LEFT:
        add_dep(s, s.i, s.top, t.label)
        pop_stack(s)
    elif t.move == RIGHT:
        add_dep(s, s.top, s.i, t.label)
        push_stack(s)
    # Left-structured boundaries atm.
    #elif move == BREAK:
    #    assert s.stack_len != 0
    #    assert s.top != 0
    #    if s.parse[s.top].head == 0:
    #        add_dep(s, s.n, s.top, self.root_label)
    #    elif s.parse[s.stack[0]].head == 0:
    #        add_dep(s, s.n, s.stack[0], self.root_label)
    #    else:
    #        raise StandardError
    #    #s.segment = True
    #    s.parse[s.top].is_break = True
    #    pop_stack(s)
    elif t.move == EDIT:
        if s.parse[s.top].head != 0:
            del_r_child(s, s.parse[s.top].head)
        edited = pop_stack(s)
        while s.parse[edited].l_valency:
            child = get_l(s, edited)
            del_l_child(s, edited)
            s.second = s.top
            s.top = child
            s.stack[s.stack_len] = child
            s.stack_len += 1
        end = edited
        while s.parse[end].r_valency:
            end = get_r(s, end)
        
        for i in range(edited, end + 1):
            s.parse[i].head = i
            # TODO
            #s.parse[i].label = self.erase_label
            s.parse[i].is_edit = True
    else:
        print t.clas
        print t.move
        print t.label
        print s.is_finished
        raise StandardError(t.clas)
    if s.i >= (s.n - 1):
        s.at_end_of_buffer = True
    if s.at_end_of_buffer and s.stack_len == 0:
        s.is_finished = True


cdef size_t get_nr_moves(list left_labels, list right_labels):
    return SHIFT + REDUCE + EDIT + BREAK + len(left_labels) + len(right_labels)


cdef int fill_moves(list left_labels, list right_labels, Transition* moves):
    cdef size_t i = 0
    moves[i].move = SHIFT; i += 1
    moves[i].move = REDUCE; i += 1
    moves[i].move = EDIT; i += 1
    moves[i].move = BREAK; i += 1
    for label in left_labels:
        moves[i].move = LEFT; moves[i].label = label; i += 1
    for label in right_labels:
        moves[i].move = LEFT; moves[i].label = label; i += 1
    for clas in range(i):
        moves[clas].clas = clas
        moves[clas].score = 0
        moves[clas].cost = 0
        moves[clas].is_valid = True

