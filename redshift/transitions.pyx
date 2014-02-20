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
    BOUNDARY
    N_MOVES


cdef transition_to_str(State* s, size_t move, label, object tokens):
    tokens = tokens + ['<end>']
    if move == SHIFT:
        return u'%s-->%s' % (tokens[s.i], tokens[s.top])
    elif move == REDUCE:
        if s.heads[s.top] == 0:
            return u'%s(%s)!!' % (tokens[s.second], tokens[s.top])
        return u'%s/%s' % (tokens[s.top], tokens[s.second])
    else:
        if move == LEFT:
            head = s.i
            child = s.top
        else:
            head = s.top
            child = s.i if s.i < len(tokens) else 0
        return u'%s(%s)' % (tokens[head], tokens[child])


cdef inline bint can_shift(State* s):
    return not s.at_end_of_buffer and not s.segment

cdef inline bint can_right(State* s):
    return s.stack_len and not s.at_end_of_buffer and not s.segment

cdef inline bint can_reduce(State* s, bint repairs):
    return s.stack_len and (s.heads[s.top] or (repairs and s.stack_len >= 2))

cdef inline bint can_left(State* s, bint repairs):
    return s.stack_len and not s.segment and (s.heads[s.top] == 0 or repairs)

cdef inline bint can_edit(State* s, bint edit):
    return s.stack_len and not s.segment and edit

cdef inline bint can_segment(State* s, size_t* moves, bint sbd):
    if not sbd:
        return False
    elif s.at_end_of_buffer:
        return False
    elif s.segment:
        return False
    elif not s.stack_len:
        return False
    elif not s.t:
        return False
    elif moves[s.history[s.t-1]] != SHIFT and moves[s.history[s.t-1]] != RIGHT:
        return False
    elif nr_headless(s) != 1:
        return False
    else:
        return True

cdef class TransitionSystem:
    def __cinit__(self, allow_reattach=False, allow_reduce=False, use_edit=False,
                  use_sbd=True):
        self.use_edit = use_edit
        self.use_sbd = use_sbd
        self.n_labels = MAX_LABELS
        self.n_tags = MAX_TAGS
        self.allow_reattach = allow_reattach
        self.allow_reduce = allow_reduce
        self.nr_class = 0
        max_classes = 2 + self.n_labels + self.n_labels + self.n_tags
        self.max_class = max_classes
        self._costs = <int*>calloc(max_classes, sizeof(int))
        self.labels = <size_t*>calloc(max_classes, sizeof(size_t))
        self.moves = <size_t*>calloc(max_classes, sizeof(size_t))
        self.l_classes = <size_t*>calloc(MAX_LABELS, sizeof(size_t))
        self.r_classes = <size_t*>calloc(MAX_LABELS, sizeof(size_t))
        self.s_id = 0
        self.d_id = self.s_id + 1
        self.e_id = self.d_id + 1
        self.b_id = self.e_id + 1
        self.l_start = self.b_id + 1
        self.l_end = 0
        self.r_start = self.l_start + 1
        self.r_end = 0
        self.counter = 0
        self.erase_label = index.hashes.encode_label('erased')
        self.root_label = index.hashes.encode_label('ROOT')
        # TODO: Clean this up, or rename them or something
        self.left_labels = []
        self.right_labels = []

    def set_labels(self, tags, left_labels, right_labels):
        self.n_tags = <size_t>max(tags)
        label_idx = index.hashes.reverse_label_index()
        self.left_labels = [label_idx[label] for label in sorted(left_labels)]
        self.right_labels = [label_idx[label] for label in sorted(right_labels)]
        self.labels[self.s_id] = 0
        self.labels[self.d_id] = 0
        self.labels[self.e_id] = 0
        self.labels[self.b_id] = 0
        self.moves[self.s_id] = <size_t>SHIFT
        self.moves[self.d_id] = <size_t>REDUCE
        self.moves[self.e_id] = <size_t>EDIT
        self.moves[self.b_id] = <size_t>BOUNDARY
        clas = self.l_start
        for label in left_labels:
            self.moves[clas] = <size_t>LEFT
            self.labels[clas] = label
            self.l_classes[label] = clas
            clas += 1
        self.l_end = clas
        self.r_start = clas
        for label in right_labels:
            self.moves[clas] = <size_t>RIGHT
            self.labels[clas] = label
            self.r_classes[label] = clas
            clas += 1
        self.r_end = clas
        self.nr_class = clas
        return clas, len(set(list(left_labels) + list(right_labels)))
        
    cdef int transition(self, size_t clas, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc, move, label, end
        cdef int idx
        if s.stack_len >= 1:
            assert s.top != 0
        assert not s.is_finished
        move = self.moves[clas]
        label = self.labels[clas]
        s.history[s.t] = clas
        s.t += 1 
        if move == SHIFT:
            push_stack(s)
        elif move == REDUCE:
            if s.heads[s.top] == 0:
                assert self.allow_reduce
                assert s.second != 0
                assert s.second < s.top
                add_dep(s, s.second, s.top, s.guess_labels[s.top])
            pop_stack(s)
        elif move == LEFT:
            child = pop_stack(s)
            if s.heads[child] != 0:
                del_r_child(s, s.heads[child])
            head = s.i
            add_dep(s, head, child, label)
        elif move == RIGHT:
            child = s.i
            head = s.top
            add_dep(s, head, child, label)
            push_stack(s)
        # Left-structured boundaries atm.
        elif move == BOUNDARY:
            assert s.stack_len != 0
            assert s.top != 0
            if s.heads[s.top] == 0:
                add_dep(s, s.n, s.top, self.root_label)
            elif s.heads[s.stack[0]] == 0:
                add_dep(s, s.n, s.stack[0], self.root_label)
            else:
                raise StandardError
            s.segment = True
            s.sbd[s.top] = 1
            pop_stack(s)
        elif move == EDIT:
            if s.heads[s.top] != 0:
                del_r_child(s, s.heads[s.top])
            s.heads[s.top] = s.top
            s.labels[s.top] = self.erase_label
            edited = pop_stack(s)
            while s.l_valencies[edited]:
                child = get_l(s, edited)
                del_l_child(s, edited)
                s.second = s.top
                s.top = child
                s.stack[s.stack_len] = child
                s.stack_len += 1
            end = edited
            while s.r_valencies[end]:
                end = get_r(s, end)
            for i in range(edited, end + 1):
                s.heads[i] = i
                s.labels[i] = self.erase_label
        else:
            print clas
            print move
            print label
            print s.is_finished
            raise StandardError(clas)
        if s.i >= (s.n - 1):
            s.at_end_of_buffer = True
        if s.at_end_of_buffer and s.stack_len == 0:
            s.is_finished = True
  
    cdef int* get_costs(self, State* s, size_t* tags, size_t* heads,
                        size_t* labels, bint* edits, size_t* sbd) except NULL:
        cdef size_t i
        cdef int* costs = self._costs
        for i in range(self.nr_class):
            costs[i] = -1
        if s.is_finished:
            return costs
        costs[self.s_id] = self.s_cost(s, heads, labels, edits, sbd)
        costs[self.d_id] = self.d_cost(s, heads, labels, edits, sbd)
        costs[self.e_id] = self.e_cost(s, heads, labels, edits, sbd)
        costs[self.b_id] = self.b_cost(s, heads, labels, edits, sbd)
        cdef int r_cost = self.r_cost(s, heads, labels, edits, sbd)
        self._label_costs(self.r_start, self.r_end, labels[s.i], heads[s.i] == s.top,
                          r_cost, costs)
        cdef int l_cost = self.l_cost(s, heads, labels, edits, sbd)
        self._label_costs(self.l_start, self.l_end, labels[s.top],
                          heads[s.top] == s.i, l_cost, costs)
        for i in range(self.nr_class):
            if costs[i] == 0:
                break
        else:
            print 'Conditions:', s.stack_len, s.n, s.at_end_of_buffer, s.segment
            print 'Top, i', s.top - 1, s.i - 1
            print 'Head set:', s.heads[s.top] - 1 if s.heads[s.top] != 0 else 0
            print 'Gold heads:', heads[s.top] - 1, heads[s.i] - 1
            print 'SBD', sbd[s.top], sbd[s.i]
            print 'Edits', edits[s.top], edits[s.i]
            print l_cost
            print nr_headless(s)
            print costs[self.b_id]
            raise StandardError
        return costs

    cdef int _label_costs(self, size_t start, size_t end, size_t label, bint add,
                          int c, int* costs) except -1:
        if c == -1:
            return 0
        cdef size_t i
        for i in range(start, end):
            costs[i] = c
            if add and self.labels[i] != label:
                costs[i] += 1

    cdef int fill_valid(self, State* s, int* valid) except -1:
        cdef size_t i
        for i in range(self.nr_class):
            valid[i] = -1
        if s.is_finished:
            return 0

        valid[self.s_id] = 0 if can_shift(s) else -1
        valid[self.d_id] = 0 if can_reduce(s, self.allow_reduce) else -1
        valid[self.e_id] = 0 if can_edit(s, self.use_edit) else -1
        valid[self.b_id] = 0 if can_segment(s, self.moves, self.use_sbd) else -1
        cdef int leftable = 0 if can_left(s, self.allow_reattach) else -1
        for i in range(self.l_start, self.l_end):
            valid[i] = leftable
        cdef int rightable = 0 if can_right(s) else -1
        for i in range(self.r_start, self.r_end):
            valid[i] = rightable
        for i in range(self.nr_class):
            if valid[i] == 0:
                break
        else:
            raise StandardError

    cdef int break_tie(self, State* s, size_t* tags, size_t* heads,
                       size_t* labels, bint* edits, size_t* sbd) except -1:
        cdef bint can_push = not s.at_end_of_buffer
        cdef bint can_pop = s.top != 0
        if can_push and not can_pop:
            return self.s_id
        elif can_push and heads[s.i] == s.top:
            return self.r_classes[labels[s.i]]
        elif heads[s.top] == s.i and (self.allow_reattach or s.heads[s.top] == 0):
            return self.l_classes[labels[s.top]]
        elif self.d_cost(s, heads, labels, edits, sbd) == 0:
            return self.d_id
        elif can_push and self.s_cost(s, heads, labels, edits, sbd) == 0:
            return self.s_id
        else:
            return self.nr_class + 1

    cdef int s_cost(self, State *s, size_t* heads, size_t* labels, bint* edits,
                    size_t* sbd):
        if not can_shift(s):
            return -1
        cdef int cost = 0
        if s.stack_len < 1:
            return 0
        if can_segment(s, self.moves, self.use_sbd):
            cost += sbd[s.top] != sbd[s.i]
        if self.use_edit and edits[s.i]:
            return cost
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int r_cost(self, State *s, size_t* heads, size_t* labels, bint* edits,
                    size_t* sbd):
        if not can_right(s):
            return -1
        cdef int cost = 0
        cost += sbd[s.top] != sbd[s.i]
        if self.use_edit and edits[s.top] and not edits[s.i]:
            return 1
        if self.use_edit and edits[s.i]:
            return cost
        if heads[s.i] == s.top:
            return cost
        elif not self.use_sbd and heads[s.i] == s.heads[s.top] == self.root_label:
            return cost
        cost += has_head_in_buffer(s, s.i, heads)
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        cost += sbd[s.top] != sbd[s.i]
        return cost

    cdef int d_cost(self, State *s, size_t* heads, size_t* labels, bint* edits,
                    size_t* sbd):
        if not can_reduce(s, self.allow_reduce):
            return -1
        cdef int cost = 0
        if s.segment:
            return 0
        if can_segment(s, self.moves, self.use_sbd):
            cost += sbd[s.top] != sbd[s.i]
        cost += has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach:
            cost += has_head_in_buffer(s, s.top, heads)
        if self.use_edit and edits[s.top] and not edits[s.heads[s.top]]:
            cost += 1
        return cost

    cdef int l_cost(self, State *s, size_t* heads, size_t* labels, bint* edits,
                    size_t* sbd) except -9000:
        if not can_left(s, self.allow_reattach):
            return -1
        cdef int cost = 0
        if can_segment(s, self.moves, self.use_sbd):
            cost += sbd[s.top] != sbd[s.i]
        # This would form a dep between an edit and non-edit word
        if self.use_edit and edits[s.top] and not edits[s.i]:
            cost += 1
        elif self.use_edit and edits[s.i]:
            return cost
        if heads[s.top] == s.i:
            return cost
        elif not self.use_sbd and heads[s.top] == heads[s.i] == self.root_label:
            return cost
        cost += has_head_in_buffer(s, s.top, heads)
        cost += has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach and heads[s.top] == s.heads[s.top]:
            cost += 1
        if self.allow_reduce and heads[s.top] == s.second:
            cost += 1
        return cost

    cdef int b_cost(self, State *s, size_t* heads, size_t* labels, bint* edits,
                    size_t* sbd):
        if not can_segment(s, self.moves, self.use_sbd):
            return -1
        if sbd[s.top] == sbd[s.i]:
            return 1
        else:
            return 0

    cdef int e_cost(self, State *s, size_t* heads, size_t* labels, bint* edits,
                    size_t* sbd):
        if not can_edit(s, self.use_edit):
            return -1
        cdef int cost = 0
        if can_segment(s, self.moves, self.use_sbd):
            cost += sbd[s.top] != sbd[s.i]
        cost += not edits[s.top]
        return cost
