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
    REDUCE
    LEFT
    RIGHT
    EDIT
    BREAK
    N_MOVES


cdef inline bint can_shift(State* s):
    return not s.at_end_of_buffer


cdef inline bint can_right(State* s):
    return s.stack_len and not s.at_end_of_buffer


cdef inline bint can_reduce(State* s):
    return s.stack_len and s.parse[s.top].head


cdef inline bint can_left(State* s):
    return s.stack_len and s.parse[s.top].head == 0

cdef bint USE_EDIT = False
cdef inline bint can_edit(State* s):
    return USE_EDIT and s.stack_len

cdef bint USE_BREAK = False
cdef bint STRICT_BREAK = True
cdef inline bint can_break(State* s):
    if not USE_BREAK:
        return False
    if s.at_end_of_buffer:
        return False
    elif not s.stack_len:
        return False
    elif STRICT_BREAK and (s.parse[s.i].l_valency != 0 or s.parse[s.top].r_valency != 0):
        return False
    elif nr_headless(s) != 1:
        return False
    else:
        return True


cdef int shift_cost(State *s, Token* gold):
    cdef int cost = 0
    if s.stack_len < 1:
        return 0
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].is_edit:
        return cost
    cost += has_child_in_stack(s, s.i, gold)
    cost += has_head_in_stack(s, s.i, gold)
    return cost


cdef int right_cost(State *s, Token* gold):
    cdef int cost = 0
    if gold[s.top].is_edit and not gold[s.i].is_edit:
        return 1
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    if gold[s.i].is_edit:
        return cost
    if gold[s.i].head == s.top:
        return cost
    if not USE_BREAK and gold[s.top].head == gold[s.i].head == (s.n - 1):
        return cost
    cost += has_head_in_buffer(s, s.i, gold)
    cost += has_child_in_stack(s, s.i, gold)
    cost += has_head_in_stack(s, s.i, gold)
    return cost


cdef int reduce_cost(State *s, Token* gold):
    cdef int cost = 0
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    cost += has_child_in_buffer(s, s.top, gold)
    if gold[s.top].is_edit and not gold[s.parse[s.top].head].is_edit:
        cost += 1
    return cost


cdef int left_cost(State *s, Token* gold) except -9000:
    cdef int cost = 0
    if can_break(s):
        cost += gold[s.top].sent_id != gold[s.i].sent_id
    # This would form a dep between an edit and non-edit word
    if gold[s.top].is_edit and not gold[s.i].is_edit:
         cost += 1
    elif gold[s.i].is_edit:
        return cost
    if gold[s.top].head == s.i:
        return cost
    if not USE_BREAK and gold[s.top].head == gold[s.i].head == (s.n - 1):
        return cost
    cost += has_head_in_buffer(s, s.top, gold)
    cost += has_child_in_buffer(s, s.top, gold)
    #if self.allow_reattach and gold[s.top].head == s.parse[s.top].head:
    #    cost += 1
    #if self.allow_reduce and gold[s.top].head == s.second:
    #    cost += 1
    return cost


cdef int break_cost(State *s, Token* gold):
    # What happens if we're at a boundary, the word on top of the stack is
    # disfluent, and its leftward children aren't? Note that the leftward childrens'
    # cost _must_ be sunk; their head has to have been off to their left.
    # But we have to choose between:
    # - Get the Edit right, return the children to the stack. Subsequently,
    # we will either Edit the children, or Left-Arc them, which would mean
    # getting the utterance boundary wrong.
    # - Get the Edit wrong, by applying Break here, but get the sentence
    # boundary right
    # In order to make the oracle work, we'll choose getting the Edit right.
    # We'll not have the oracle refer to sentence boundaries specifically.
    # If costs are sunk by having a word's head already incorrect, we don't
    # try to enforce the boundary as well. We train for syntax first.
    return gold[s.top].is_edit or gold[s.top].sent_id == gold[s.i].sent_id


cdef int edit_cost(State *s, Token* gold):
    cdef int cost = 0
    # TODO: Is this good? I suspect not!
    #if can_break(s):
    #    cost += gold[s.top].sent_id != gold[s.i].sent_id
    cost += not gold[s.top].is_edit
    return cost


cdef int fill_valid(State* s, Transition* classes, size_t n) except -1:
    cdef bint[N_MOVES] valid
    valid[SHIFT] = can_shift(s)
    valid[REDUCE] = can_reduce(s)
    valid[LEFT] = can_left(s)
    valid[RIGHT] = can_right(s)
    valid[EDIT] = can_edit(s)
    valid[BREAK] = can_break(s)
    for i in range(n):
        classes[i].is_valid = valid[classes[i].move]
    for i in range(n):
        if classes[i].is_valid:
            break
    else:
        print s.top
        print s.i
        print s.parse[s.top].head
        print s.n
        print s.at_end_of_buffer
        print s.is_finished
        raise StandardError


cdef int fill_costs(State* s, Transition* classes, size_t n, Token* gold) except -1:
    cdef int[N_MOVES] costs
    costs[SHIFT] = shift_cost(s, gold) if can_shift(s) else -1
    costs[REDUCE] = reduce_cost(s, gold) if can_reduce(s) else -1
    costs[LEFT] = left_cost(s, gold) if can_left(s) else -1
    costs[RIGHT] = right_cost(s, gold) if can_right(s) else -1
    costs[EDIT] = edit_cost(s, gold) if can_edit(s) else -1
    costs[BREAK] = break_cost(s, gold) if can_break(s) else -1
    #print costs[SHIFT], costs[REDUCE], costs[LEFT], costs[RIGHT], costs[EDIT], costs[BREAK]
    for i in range(n):
        classes[i].cost = costs[classes[i].move]
        if classes[i].move == LEFT and classes[i].cost == 0:
            classes[i].cost += gold[s.top].label != classes[i].label
        elif classes[i].move == RIGHT and costs[RIGHT] == 0:
            classes[i].cost = gold[s.i].label != classes[i].label


cdef int transition(Transition* t, State *s) except -1:
    if s.stack_len >= 1:
        assert s.top != 0
    assert not s.is_finished
    s.history[s.m] = t[0]
    s.m += 1 
    cdef size_t erase_label
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
    elif t.move == BREAK:
        assert s.stack_len != 0
        assert s.top != 0
        s.parse[s.i].sent_id = s.parse[s.top].sent_id + 1
        root_label = index.hashes.encode_label('ROOT')
        while s.stack_len:
            if s.parse[s.top].head == 0:
                add_dep(s, s.n, s.top, root_label)
            pop_stack(s)
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
        erase_label = index.hashes.encode_label('erased')
        while s.parse[end].r_valency:
            end = get_r(s, end)
        
        for i in range(edited, end + 1):
            s.parse[i].head = i
            # TODO
            s.parse[i].label = erase_label
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


cdef size_t get_nr_moves(list left_labels, list right_labels,
                         bint use_edit, bint use_break):
    global USE_BREAK, USE_EDIT
    USE_BREAK = use_break
    USE_EDIT = use_edit
    return 1 + 1 + use_edit + use_break + len(left_labels) + len(right_labels)


cdef int fill_moves(list left_labels, list right_labels, bint use_edit, bint use_break,
                    Transition* moves):
    cdef size_t i = 0
    moves[i].move = SHIFT; i += 1
    moves[i].move = REDUCE; i += 1
    if use_edit:
        moves[i].move = EDIT; i += 1
    if use_break:
        moves[i].move = BREAK; i += 1
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

