# cython: profile=True
from _state cimport *
from libc.stdlib cimport malloc, calloc, free
import redshift.io_parse

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
    def __cinit__(self, object tags, object labels, allow_reattach=False,
                  allow_reduce=False):
        self.assign_pos = False
        self.n_labels = len(labels)
        self.n_tags = max(tags)
        self.py_labels = labels
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
        # TODO: Fix this
        self.erase_label = redshift.io_parse.STR_TO_LABEL.get('erased', 9000)
        self.counter = 0

    def set_labels(self, tags, left_labels, right_labels):
        self.n_tags = <size_t>max(tags)
        self.left_labels = [self.py_labels[l] for l in sorted(left_labels)]
        self.right_labels = [self.py_labels[l] for l in sorted(right_labels)]
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
        return clas
        
    cdef int transition(self, size_t clas, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc, move, label
        cdef int idx
        if s.stack_len >= 1:
            assert s.top != 0
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
        #elif move == ASSIGN_POS:
        #    s.tags[s.i + 1] = label
        else:
            print clas
            print move
            print label
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
        cdef size_t last_move = self.moves[s.history[s.t - 1]] if s.t != 0 else SHIFT
        if self.assign_pos and (s.i < (s.n - 2)) and \
          (last_move == SHIFT or last_move == RIGHT):
            for i in range(self.p_start, self.p_end):
                costs[i] = 1
            costs[self.p_classes[tags[s.i + 1]]] = 0
            return costs
        costs[self.s_id] = self.s_cost(s, heads, labels, edits)
        costs[self.d_id] = self.d_cost(s, heads, labels, edits)
        costs[self.e_id] = self.e_cost(s, heads, labels, edits)
        r_cost = self.r_cost(s, heads, labels, edits)
        if r_cost != -1:
            for i in range(self.r_start, self.r_end):
                costs[i] = r_cost
                if heads[s.i] == s.top and self.labels[i] != labels[s.i]:
                    costs[i] += 1
        l_cost = self.l_cost(s, heads, labels, edits)
        if l_cost != -1:
            for i in range(self.l_start, self.l_end):
                costs[i] = l_cost
                if heads[s.top] == s.i and self.labels[i] != labels[s.top]:
                    costs[i] += 1
            # Add an additional penalty for using the ROOT label inappropriately,
            # as it signals SBD
            #if labels[s.top] != 1:
            #    costs[self.l_classes[1]] += 1
        return costs

    cdef int fill_valid(self, State* s, int* valid) except -1:
        cdef size_t i
        for i in range(self.nr_class):
            valid[i] = -1
        cdef size_t last_move
        if s.t != 0:
            last_move = self.moves[s.history[s.t - 1]]
        else:
            last_move = SHIFT
        if self.assign_pos and (s.i < (s.n - 2)) and \
          (last_move == SHIFT or last_move == RIGHT):
            for i in range(self.p_start, self.p_end):
                valid[i] = 0
            return 0
        if not s.at_end_of_buffer:
            valid[self.s_id] = 0
            if s.stack_len < 1:
                return 0
            if not has_root_child(s, s.i):
                for i in range(self.r_start, self.r_end):
                    valid[i] = 0
        if s.top != 0:
            valid[self.e_id] = 0
        if s.stack_len >= 1:
            if s.heads[s.top] != 0 or (s.stack_len >= 2 and self.allow_reattach):
                valid[self.d_id] = 0
            if self.allow_reattach or s.heads[s.top] == 0:
                #if has_root_child(s, s.i) or has_root_child(s, s.top):
                #    valid[self.l_classes[1]] = 0
                #else:
                if not has_root_child(s, s.i):
                    for i in range(self.l_start, self.l_end):
                        valid[i] = 0

    cdef int fill_static_costs(self, State* s, size_t* tags, size_t* heads,
                               size_t* labels, bint* edits, int* costs) except -1:
        cdef size_t oracle = self.break_tie(s, tags, heads, labels, edits)
        cdef int cost = s.cost
        cdef size_t i
        for i in range(self.nr_class):
            costs[i] = cost + (i != oracle)

    cdef int break_tie(self, State* s, size_t* tags, size_t* heads,
                       size_t* labels, bint* edits) except -1:
        cdef size_t last_move
        if s.t != 0:
            last_move = self.moves[s.history[s.t - 1]]
        else:
            last_move = SHIFT
        if self.assign_pos and (s.i < (s.n - 2)) and \
          (last_move == SHIFT or last_move == RIGHT):
            return self.p_classes[tags[s.i + 1]]
        if s.stack_len < 1 and not s.at_end_of_buffer:
            return self.s_id
        elif not s.at_end_of_buffer and heads[s.i] == s.top:
            return self.r_classes[labels[s.i]]
        elif heads[s.top] == s.i and (self.allow_reattach or s.heads[s.top] == 0):
            if edits[s.top] and not edits[heads[s.top]]:
                return self.e_id
            else:
                return self.l_classes[labels[s.top]]
        elif self.d_cost(s, heads, labels, edits) == 0:
            if edits[s.top] and not edits[heads[s.top]]:
                return self.e_id
            else:
                return self.d_id
        elif not s.at_end_of_buffer and self.s_cost(s, heads, labels, edits) == 0:
            return self.s_id
        elif s.top != 0 and edits[s.top]:
            return self.e_id
        else:
            return self.nr_class + 1

    cdef int s_cost(self, State *s, size_t* heads, size_t* labels, bint* edits):
        cdef int cost = 0
        cdef size_t i, stack_i
        if s.at_end_of_buffer:
            return -1
        if edits[s.i]:
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
        if edits[s.top] and not edits[s.i]:
            return 1
        if edits[s.i]:
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
        # TODO; What's this?
        #if heads[s.top] == 0 and (s.stack_len == 2 or not self.allow_reattach):
        #    cost += 1
        cost += has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach:
            cost += has_head_in_buffer(s, s.top, heads)
            if cost == 0 and s.second == 0:
                return -1
        if edits[s.top] and not edits[s.heads[s.top]]:
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
        # This will form a dep between an edit and non-edit word
        if edits[s.top] and not edits[s.i]:
            return 1
        elif edits[s.top]:
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
        if s.top == 0:
            return -1
        if edits[s.top]:
            return 0
        else:
            return 1


