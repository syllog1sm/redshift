# cython: profile=True
from _state cimport *
from libc.stdlib cimport malloc, calloc, free
import redshift.io_parse
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
    ASSIGN_POS
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

cdef class TransitionSystem:
    def __cinit__(self, allow_reattach=False, allow_reduce=False, use_edit=False):
        self.assign_pos = False
        self.use_edit = use_edit
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
        self.l_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.r_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.p_classes = <size_t*>calloc(self.n_tags, sizeof(size_t))
        self.s_id = 0
        self.d_id = self.s_id + 1
        self.e_id = self.d_id + 1
        self.l_start = self.e_id + 1
        self.l_end = 0
        self.r_start = self.l_start + 1
        self.r_end = 0
        self.p_start = self.r_start + 1
        self.p_end = 0
        self.counter = 0
        self.erase_label = index.hashes.encode_label('erased')
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
        self.moves[self.s_id] = <size_t>SHIFT
        self.moves[self.d_id] = <size_t>REDUCE
        self.moves[self.e_id] = <size_t>EDIT
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
        cdef size_t tag
        if self.assign_pos:
            self.p_start = clas
            for tag in sorted(tags):
                self.moves[clas] = <size_t>ASSIGN_POS
                self.labels[clas] = tag
                self.p_classes[tag] = clas
                clas += 1
            self.p_end = clas
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
        #elif move == ASSIGN_POS:
        #    s.tags[s.i + 1] = label
        else:
            print clas
            print move
            print label
            print s.is_finished
            raise StandardError(clas)
        if s.i == (s.n - 1):
            s.at_end_of_buffer = True
        if s.at_end_of_buffer and s.stack_len == 0:
            s.is_finished = True
  
    cdef int* get_costs(self, State* s, size_t* tags, size_t* heads,
                        size_t* labels, bint* edits) except NULL:
        if s.stack_len >= 1:
            assert s.top != 0, s.stack_len
        cdef size_t i
        cdef int* costs = self._costs
        for i in range(self.nr_class):
            costs[i] = -1
        if s.is_finished:
            return costs
        p_cost = self.p_cost(s)
        self._label_costs(self.p_start, self.p_end, tags[s.i + 1], True, p_cost, costs)
        costs[self.s_id] = self.s_cost(s, heads, labels, edits)
        costs[self.d_id] = self.d_cost(s, heads, labels, edits)
        costs[self.e_id] = self.e_cost(s, heads, labels, edits)
        cdef int r_cost = self.r_cost(s, heads, labels, edits)
        self._label_costs(self.r_start, self.r_end, labels[s.i], heads[s.i] == s.top,
                          r_cost, costs)
        cdef int l_cost = self.l_cost(s, heads, labels, edits)
        self._label_costs(self.l_start, self.l_end, labels[s.top],
                          heads[s.top] == s.i, l_cost, costs)
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
        if self.p_cost(s) != -1:
            for i in range(self.p_start, self.p_end):
                valid[i] = 0
            return 0
        if s.is_finished:
            return 0
        cdef bint can_push = not s.at_end_of_buffer
        cdef bint can_pop = s.top != 0
        if can_push:
            valid[self.s_id] = 0
        if can_pop and (s.heads[s.top] != 0 or (self.allow_reduce and s.stack_len >= 2)):
            valid[self.d_id] = 0
        if can_pop and self.use_edit:
            valid[self.e_id] = 0
        if can_push and can_pop:
            for i in range(self.r_start, self.r_end):
                valid[i] = 0
        if can_pop and (s.heads[s.top] == 0 or self.allow_reattach):
            for i in range(self.l_start, self.l_end):
                valid[i] = 0

    cdef int fill_static_costs(self, State* s, size_t* tags, size_t* heads,
                               size_t* labels, bint* edits, int* costs) except -1:
        cdef size_t oracle = self.break_tie(s, tags, heads, labels, edits)
        cdef size_t i
        for i in range(self.nr_class):
            costs[i] = i != oracle

    cdef int break_tie(self, State* s, size_t* tags, size_t* heads,
                       size_t* labels, bint* edits) except -1:
        if self.p_cost(s) != -1:
            return self.p_classes[tags[s.i + 1]]
        cdef bint can_push = not s.at_end_of_buffer
        cdef bint can_pop = s.top != 0
        if can_push and not can_pop:
            return self.s_id
        elif can_push and heads[s.i] == s.top:
            return self.r_classes[labels[s.i]]
        elif heads[s.top] == s.i and (self.allow_reattach or s.heads[s.top] == 0):
            return self.l_classes[labels[s.top]]
        elif self.d_cost(s, heads, labels, edits) == 0:
            return self.d_id
        elif can_push and self.s_cost(s, heads, labels, edits) == 0:
            return self.s_id
        else:
            return self.nr_class + 1

    cdef int s_cost(self, State *s, size_t* heads, size_t* labels, bint* edits):
        cdef int cost = 0
        cdef size_t i, stack_i
        if s.at_end_of_buffer:
            return -1
        if self.use_edit and edits[s.i]:
            return 0
        if s.stack_len < 1:
            return 0
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int r_cost(self, State *s, size_t* heads, size_t* labels, bint* edits):
        cdef int cost = 0
        cdef size_t i, buff_i, stack_i
        if s.at_end_of_buffer:
            return -1
        if s.stack_len < 1:
            return -1
        if has_root_child(s, s.i):
            return -1
        if self.use_edit and edits[s.top] and not edits[s.i]:
            return 1
        if self.use_edit and edits[s.i]:
            return 0
        if heads[s.i] == s.top:
            return 0
        cost += has_head_in_buffer(s, s.i, heads)
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int d_cost(self, State *s, size_t* heads, size_t* labels, bint* edits):
        cdef int cost = 0
        if s.heads[s.top] == 0 and not self.allow_reduce:
            return -1
        if s.stack_len < 1:
            return -1
        cost += has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach:
            cost += has_head_in_buffer(s, s.top, heads)
            if cost == 0 and s.second == 0:
                return -1
        if self.use_edit and edits[s.top] and not edits[s.heads[s.top]]:
            cost += 1
        return cost

    cdef int l_cost(self, State *s, size_t* heads, size_t* labels, bint* edits) except -9000:
        cdef size_t buff_i, i
        cdef int cost = 0
        if s.stack_len < 1:
            return -1
        if s.heads[s.top] != 0 and not self.allow_reattach:
            return -1
        if has_root_child(s, s.i):
            return -1
        # This would form a dep between an edit and non-edit word
        if self.use_edit and edits[s.top] and not edits[s.i]:
            return 1
        elif self.use_edit and edits[s.i]:
            return 0
        if heads[s.top] == s.i:
            return 0
        cost +=  has_head_in_buffer(s, s.top, heads)
        cost +=  has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach and heads[s.top] == s.heads[s.top]:
            cost += 1
        if self.allow_reduce and heads[s.top] == s.second:
            cost += 1
        return cost
    
    cdef int e_cost(self, State *s, size_t* heads, size_t* labels, bint* edits):
        if not self.use_edit:
            return -1
        if s.top == 0:
            return -1
        if edits[s.top]:
            return 0
        else:
            return 1

    cdef int p_cost(self, State* s):
        if not self.assign_pos:
            return -1
        if s.i >= (s.n - 2):
            return -1
        if s.t == 0:
            return 0
        cdef size_t last_move = self.moves[s.history[s.t - 1]]
        if last_move == SHIFT or last_move == RIGHT:
            return 0
        return -1

