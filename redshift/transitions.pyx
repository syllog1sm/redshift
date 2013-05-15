# cython: profile=True
from _state cimport *
from libc.stdlib cimport malloc, calloc, free

cdef enum:
    ERR
    SHIFT
    REDUCE
    LEFT
    RIGHT
    _n_moves

DEF N_MOVES = 5
assert N_MOVES == _n_moves, "Set N_MOVES compile var to %d" % _n_moves


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

    def __cinit__(self, object labels, allow_reattach=False,
                  allow_reduce=False):
        self.n_labels = len(labels)
        self.py_labels = labels
        self.allow_reattach = allow_reattach
        self.allow_reduce = allow_reduce
        self.nr_class = 0
        max_classes = N_MOVES * len(labels)
        self.max_class = max_classes
        self._costs = <int*>calloc(max_classes, sizeof(int))
        self.labels = <size_t*>calloc(max_classes, sizeof(size_t))
        self.moves = <size_t*>calloc(max_classes, sizeof(size_t))
        self.l_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.r_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.s_id = 0
        self.d_id = 1
        self.l_start = 2
        self.l_end = 0
        self.r_start = 3
        self.r_end = 0

    def set_labels(self, left_labels, right_labels):
        self.left_labels = [self.py_labels[l] for l in sorted(left_labels)]
        self.right_labels = [self.py_labels[l] for l in sorted(right_labels)]
        self.labels[self.s_id] = 0
        self.labels[self.d_id] = 0
        self.moves[self.s_id] = <int>SHIFT
        self.moves[self.d_id] = <int>REDUCE
        clas = self.l_start
        for label in left_labels:
            self.moves[clas] = <int>LEFT
            self.labels[clas] = label
            self.l_classes[label] = clas
            clas += 1
        self.l_end = clas
        self.r_start = clas
        for label in right_labels:
            self.moves[clas] = <int>RIGHT
            self.labels[clas] = label
            self.r_classes[label] = clas
            clas += 1
        self.r_end = clas
        self.nr_class = clas
        return clas
        
    cdef int transition(self, size_t clas, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc, move, label
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
        else:
            print move
            print label
            raise StandardError(clas)
        if s.i == (s.n - 1):
            s.at_end_of_buffer = True
        if s.at_end_of_buffer and s.stack_len == 1:
            s.is_finished = True
  
    cdef int* get_costs(self, State* s, size_t* heads, size_t* labels) except NULL:
        cdef size_t i
        cdef int* costs = self._costs
        for i in range(self.nr_class):
            costs[i] = -1
        if s.stack_len == 1 and not s.at_end_of_buffer:
            costs[self.s_id] = 0
        if not s.at_end_of_buffer:
            costs[self.s_id] = self.s_cost(s, heads, labels)
            r_cost = self.r_cost(s, heads, labels)
            if r_cost != -1:
                for i in range(self.r_start, self.r_end):
                    if heads[s.i] == s.top and self.labels[i] != labels[s.i]:
                        costs[i] = r_cost + 1
                    else:
                        costs[i] = r_cost
        if s.stack_len >= 2:
            costs[self.d_id] = self.d_cost(s, heads, labels)
            l_cost = self.l_cost(s, heads, labels)
            if l_cost != -1:
                for i in range(self.l_start, self.l_end):
                    if heads[s.top] == s.i and self.labels[i] != labels[s.top]:
                        costs[i] = l_cost + 1
                    else:
                        costs[i] = l_cost
        return costs

    cdef int fill_valid(self, State* s, int* valid) except -1:
        cdef size_t i
        for i in range(self.nr_class):
            valid[i] = -1
        if not s.at_end_of_buffer:
            valid[self.s_id] = 0
            if s.stack_len == 1:
                return 0
            if not has_root_child(s, s.i):
                for i in range(self.r_start, self.r_end):
                    valid[i] = 0
        else:
            valid[self.s_id] = -1
        if s.stack_len != 1:
            if s.heads[s.top] != 0:
                valid[self.d_id] = 0
            elif (self.allow_reattach or s.heads[s.top] == 0):
                #if has_root_child(s, s.i) or has_root_child(s, s.top):
                #    valid[self.l_classes[1]] = 0
                #else:
                if not has_root_child(s, s.i):
                    for i in range(self.l_start, self.l_end):
                        valid[i] = 0
        if s.stack_len >= 3 and self.allow_reduce:
            valid[self.d_id] = 0
            assert s.second != 0

    cdef int fill_static_costs(self, State* s, size_t* heads, size_t* labels,
                               int* costs) except -1:
        cdef size_t oracle = self.break_tie(s, heads, labels)
        cdef int cost = s.cost
        cdef size_t i
        for i in range(self.nr_class):
            costs[i] = cost + (i != oracle)

    cdef int break_tie(self, State* s, size_t* heads, size_t* labels) except -1:
        if s.stack_len == 1:
            return self.s_id
        elif not s.at_end_of_buffer and heads[s.i] == s.top:
            return self.r_classes[labels[s.i]]
        elif heads[s.top] == s.i and (self.allow_reattach or s.heads[s.top] == 0):
            return self.l_classes[labels[s.top]]
        elif self.d_cost(s, heads, labels) == 0:
            return self.d_id
        elif not s.at_end_of_buffer and self.s_cost(s, heads, labels) == 0:
            return self.s_id
        else:
            return self.nr_class + 1

    cdef int s_cost(self, State *s, size_t* heads, size_t* labels):
        cdef int cost = 0
        cdef size_t i, stack_i
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int r_cost(self, State *s, size_t* heads, size_t* labels):
        cdef int cost = 0
        cdef size_t i, buff_i, stack_i
        if has_root_child(s, s.i):
            return -1
        if heads[s.i] == s.top:
            return 0
        if not self.allow_reattach:
            cost += has_head_in_buffer(s, s.i, heads)
        cost += has_child_in_stack(s, s.i, heads)
        cost += has_head_in_stack(s, s.i, heads)
        return cost

    cdef int d_cost(self, State *s, size_t* g_heads, size_t* g_labels):
        cdef int cost = 0
        if s.heads[s.top] == 0 and not self.allow_reduce:
            return -1
        #if g_heads[s.top] == 0 and (s.stack_len == 2 or not self.allow_reattach):
        #    cost += 1
        cost += has_child_in_buffer(s, s.top, g_heads)
        if self.allow_reattach:
            cost += has_head_in_buffer(s, s.top, g_heads)
            if cost == 0 and s.second == 0:
                return -1
        return cost

    cdef int l_cost(self, State *s, size_t* heads, size_t* labels):
        cdef size_t buff_i, i
        cdef int cost = 0
        if s.heads[s.top] != 0 and not self.allow_reattach:
            return -1
        if has_root_child(s, s.i):
            return -1
        if heads[s.top] == s.i:
            return 0
        cost +=  has_head_in_buffer(s, s.top, heads)
        cost +=  has_child_in_buffer(s, s.top, heads)
        if self.allow_reattach and heads[s.top] == s.heads[s.top]:
            cost += 1
        if self.allow_reduce and heads[s.top] == s.second:
            cost += 1
        return cost


"""
cdef class ArcStandard(TransitionSystem):        
    def __cinit__(self, object labels, allow_reattach=False,
                  allow_reduce=False):
        self.n_labels = len(labels)
        self.py_labels = labels
        self.allow_reattach = allow_reattach
        self.allow_reduce = allow_reduce
        self.nr_class = 0
        max_classes = N_MOVES * len(labels)
        self.max_class = max_classes
        self._costs = <int*>calloc(max_classes, sizeof(int))
        self.labels = <size_t*>calloc(max_classes, sizeof(size_t))
        self.moves = <size_t*>calloc(max_classes, sizeof(size_t))
        self.l_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.r_classes = <size_t*>calloc(self.n_labels, sizeof(size_t))
        self.s_id = 0
        self.l_start = 1
        self.l_end = 0
        self.r_start = 0
        self.r_end = 0
        self.d_id = 0

    def set_labels(self, left_labels, right_labels):
        self.left_labels = [self.py_labels[l] for l in sorted(left_labels)]
        self.right_labels = [self.py_labels[l] for l in sorted(right_labels)]
        self.labels[self.s_id] = 0
        self.moves[self.s_id] = <int>SHIFT
        clas = self.l_start
        for label in left_labels:
            self.moves[clas] = <int>LEFT
            self.labels[clas] = label
            self.l_classes[label] = clas
            clas += 1
        self.l_end = clas
        self.r_start = clas
        for label in right_labels:
            self.moves[clas] = <int>RIGHT
            self.labels[clas] = label
            self.r_classes[label] = clas
            clas += 1
        self.r_end = clas
        self.nr_class = clas
        return clas
      

    cdef int transition(self, size_t clas, State *s) except -1:
        cdef size_t head, child, new_parent, new_child, c, gc, move, label
        move = self.moves[clas]
        label = self.labels[clas]
        s.history[s.t] = clas
        s.t += 1 
        if move == SHIFT:
            assert not s.at_end_of_buffer
            push_stack(s)
        elif move == LEFT:
            assert s.stack_len >= 3
            head = s.top
            child = s.second
            add_dep(s, head, child, label)
            s.stack_len -= 1
            s.stack[s.stack_len - 1] = s.top
            s.second = s.stack[s.stack_len - 2]
        elif move == RIGHT:
            assert s.stack_len >= 3
            child = s.top
            head = s.second
            add_dep(s, head, child, label)
            pop_stack(s)
        else:
            raise StandardError(clas)
        if s.i == (s.n - 1):
            s.at_end_of_buffer = True
        if s.at_end_of_buffer and s.stack_len == 2:
            s.is_finished = True
            add_dep(s, s.i, s.top, 0)
            pop_stack(s)

    cdef int break_tie(self, State* s, size_t* heads, size_t* labels) except -1:
        if s.stack_len <= 2 and not s.at_end_of_buffer:
            return self.s_id
        elif heads[s.top] == s.second and not has_child_in_buffer(s, s.top, heads):
            return self.r_classes[labels[s.top]]
        elif heads[s.second] == s.top:
            return self.l_classes[labels[s.second]]
        elif s.at_end_of_buffer:
            return self.nr_class + 1
        elif heads[s.i] == s.top or heads[s.top] == s.i:
            return self.s_id
        elif not has_head_in_stack(s, s.i, heads) and not has_child_in_stack(s, s.i, heads):
            return self.s_id
        else:
            return self.nr_class + 1

    cdef int* get_valid(self, State* s):
        cdef size_t i
        cdef int* valid = self._costs
        if s.stack_len >= 3:
            validity = 0
        else:
            validity = -1
        for i in range(self.r_start, self.r_end):
            valid[i] = validity
        for i in range(self.l_start, self.l_end):
            valid[i] = validity
        if s.at_end_of_buffer:
            valid[self.s_id] = -1
        else:
            valid[self.s_id] = 0
        return valid
"""
