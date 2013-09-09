from _state cimport State
from _state cimport has_head_in_buffer, has_child_in_buffer
from _state cimport has_head_in_stack, has_child_in_stack
from _fast_state cimport *

#cdef transition_to_str(State* s, size_t move, label, object tokens)


cdef class TransitionSystem:
    cdef bint use_edit
    cdef bint allow_reattach
    cdef bint allow_reduce
    cdef bint assign_pos
    cdef size_t n_labels
    cdef size_t n_tags
    cdef object py_tags
    cdef object py_labels
    cdef int* _costs
    cdef size_t* labels
    cdef size_t* moves
    cdef size_t* l_classes
    cdef size_t* r_classes
    cdef size_t* p_classes
    cdef list left_labels
    cdef list right_labels
    cdef size_t nr_class
    cdef size_t max_class
    cdef size_t s_id
    cdef size_t d_id
    cdef size_t e_id
    cdef size_t l_start
    cdef size_t l_end
    cdef size_t r_start
    cdef size_t r_end
    cdef size_t p_start
    cdef size_t p_end
    cdef size_t erase_label
    cdef size_t counter

    cdef int transition(self, size_t clas, State *s) except -1
    cdef int fill_costs(self, int* costs, size_t n0, size_t length, size_t stack_len,
                        size_t* stack, bint has_head, size_t* tags, size_t* heads,
                        size_t* labels, bint* edits) except -1
    cdef int _label_costs(self, int* costs, int c, size_t start, size_t end,
                          size_t label, bint add) except -1
    cdef int fill_static_costs(self, State* s, size_t* tags, size_t* heads,
                               size_t* labels, bint* edits, int* costs) except -1
    cdef int fill_valid(self, int* valid, bint can_push, bint has_stack,
                        bint has_head) except -1
    cdef int break_tie(self, bint can_push, bint has_head, 
                       size_t n0, size_t s0, size_t length, size_t* tags,
                       size_t* heads, size_t* labels, bint* edits) except -1
    cdef int s_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    size_t* heads, size_t* labels, bint* edits)
    cdef int r_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    size_t* heads, size_t* labels, bint* edits)
    cdef int d_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    bint has_head, size_t* heads, size_t* labels, bint* edits)
    cdef int l_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    bint has_head, size_t* heads, size_t* labels, bint* edits)
    cdef int e_cost(self, size_t n0, size_t length, size_t stack_len, size_t* stack,
                    size_t* heads, size_t* labels, bint* edits)
    cdef int p_cost(self, State *s)
