# cython: profile=True
from _fast_state cimport *
from libc.stdlib cimport malloc, calloc, free
import redshift.io_parse
import index.hashes
from _fast_state cimport *


# TODO: Link these with other compile constants
DEF MAX_TAGS = 100
DEF MAX_LABELS = 200
DEF ERASED = 99


cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    EDIT
    ASSIGN_POS
    N_MOVES


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
        self.l_classes = <size_t*>calloc(MAX_LABELS, sizeof(size_t))
        self.r_classes = <size_t*>calloc(MAX_LABELS, sizeof(size_t))
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
    
    cdef int fill_costs(self, int* costs, size_t n0, size_t length, size_t stack_len, 
                        size_t* stack, bint has_head, size_t* tags, size_t* heads,
                        size_t* labels, bint* edits) except -1:
        cdef size_t i
        cdef size_t s0 = stack[0] if stack_len >= 1 else 0
        for i in range(self.nr_class):
            costs[i] = -1
        if n0 == length - 1 and stack_len == 0:
            return 0
        #p_cost = self.p_cost(s)
        #self._label_costs(self.p_start, self.p_end, tags[n0 + 1], True, p_cost, costs)
        costs[self.s_id] = self.s_cost(n0, length, stack_len, stack, heads, labels, edits)
        costs[self.d_id] = self.d_cost(n0, length, stack_len, stack, has_head,
                                       heads, labels, edits)
        costs[self.e_id] = self.e_cost(n0, length, stack_len, stack, heads, labels, edits)
        cdef int r_cost = self.r_cost(n0, length, stack_len, stack, heads, labels, edits)
        self._label_costs(costs, r_cost, self.r_start, self.r_end, labels[n0],
                          heads[n0] == s0)
        cdef int l_cost = self.l_cost(n0, length, stack_len, stack, has_head,
                                      heads, labels, edits)
        self._label_costs(costs, l_cost, self.l_start, self.l_end, labels[s0],
                          heads[s0] == n0)

    cdef int _label_costs(self, int* costs, int c, size_t start, size_t end,
                          size_t label, bint add) except -1:
        if c == -1:
            return 0
        cdef size_t i
        for i in range(start, end):
            costs[i] = c
            if add and self.labels[i] != label:
                costs[i] += 1

    cdef int fill_valid(self, int* valid, bint can_push, bint has_stack,
                        bint has_head) except -1:
        cdef size_t i
        for i in range(self.nr_class):
            valid[i] = -1
        if not can_push and not has_stack:
            return 0
        if can_push:
            valid[self.s_id] = 0
        if has_stack and has_head:
            valid[self.d_id] = 0
        if has_stack and self.use_edit:
            valid[self.e_id] = 0
        if can_push and has_stack:
            for i in range(self.r_start, self.r_end):
                valid[i] = 0
        if has_stack and not has_head:
            for i in range(self.l_start, self.l_end):
                valid[i] = 0

    cdef int break_tie(self, bint can_push, bint has_head, 
                       size_t n0, size_t s0, size_t length, size_t* tags,
                       size_t* heads, size_t* labels, bint* edits) except -1:
        cdef bint has_stack = s0 != 0
        if can_push and not has_stack:
            return self.s_id
        elif can_push and heads[n0] == s0:
            return self.r_classes[labels[n0]]
        if heads[s0] == n0 and not has_head:
            return self.l_classes[labels[s0]]
        cdef size_t i
        if has_head and has_stack:
            for i in range(n0, length):
                if heads[i] == s0:
                    break
            else:
                return self.d_id
        if can_push:
            return self.s_id
        raise StandardError

    cdef int s_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    size_t* heads, size_t* labels, bint* edits):
        cdef size_t s0 = stack[0] if stack_len >= 1 else 0
        cdef int cost = 0
        if n0 == length - 1:
            return -1
        if self.use_edit and edits[n0]:
            return 0
        if stack_len < 1:
            return 0
        cost += has_child_in_stack(n0, stack_len, stack, heads)
        cost += has_head_in_stack(n0, stack_len, stack, heads)
        return cost

    cdef int r_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    size_t* heads, size_t* labels, bint* edits):
        cdef size_t s0 = stack[0] if stack_len >= 1 else 0
        cdef int cost = 0
        cdef size_t i, buff_i, stack_i
        if n0 == length - 1:
            return -1
        if stack_len < 1:
            return -1
        if self.use_edit and edits[s0] and not edits[n0]:
            return 1
        if self.use_edit and edits[n0]:
            return 0
        if heads[n0] == s0:
            return 0
        cost += has_head_in_buffer(n0, n0, length, heads)
        cost += has_child_in_stack(n0, stack_len, stack, heads)
        cost += has_head_in_stack(n0, stack_len, stack, heads)
        return cost

    cdef int d_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    bint has_head, size_t* heads, size_t* labels, bint* edits):
        cdef int cost = 0
        cdef size_t s0 = stack[0] if stack_len >= 1 else 0
        cdef size_t s1 = stack[1] if stack_len >= 2 else 0
        if not has_head and not self.allow_reduce:
            return -1
        if stack_len < 1:
            return -1
        cost += has_child_in_buffer(s0, n0, length, heads)
        if self.allow_reattach:
            cost += has_head_in_buffer(s0, n0, length, heads)
            if cost == 0 and s1 == 0:
                return -1
        if self.use_edit and edits[s0] and (has_head and not edits[s1]):
            cost += 1
        return cost

    cdef int l_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    bint has_head, size_t* heads, size_t* labels, bint* edits) except -9000:
        cdef size_t s0 = stack[0] if stack_len >= 1 else 0
        cdef size_t s1 = stack[1] if stack_len >= 2 else 0
        cdef size_t buff_i, i
        cdef int cost = 0
        if stack_len < 1:
            return -1
        if n0 == 0:
            return -1
        if has_head and not self.allow_reattach:
            return -1
        # This would form a dep between an edit and non-edit word
        if self.use_edit and edits[s0] and not edits[n0]:
            return 1
        elif self.use_edit and edits[n0]:
            return 0
        if heads[s0] == n0:
            return 0
        cost +=  has_head_in_buffer(s0, n0, length, heads)
        cost +=  has_child_in_buffer(s0, n0, length, heads)
        if (self.allow_reattach or self.allow_reduce) and heads[s0] == s1:
            cost += 1
        return cost
    
    cdef int e_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    size_t* heads, size_t* labels, bint* edits):
        cdef size_t s0 = stack[0] if stack_len >= 1 else 0
        if not self.use_edit:
            return -1
        if s0 == 0:
            return -1
        if edits[s0]:
            return 0
        else:
            return 1

    cdef int p_cost(self) except -9000:
        if not self.assign_pos:
            return -1
        raise NotImplementedError
