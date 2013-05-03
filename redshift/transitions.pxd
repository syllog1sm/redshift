from _state cimport *

cdef transition_to_str(State* s, size_t move, label, object tokens)


cdef class TransitionSystem:
    cdef bint allow_reattach
    cdef bint allow_reduce
    cdef size_t n_labels
    cdef object py_labels
    cdef int* _costs
    cdef size_t* labels
    cdef size_t* moves
    cdef size_t* l_classes
    cdef size_t* r_classes
    cdef list left_labels
    cdef list right_labels
    cdef size_t nr_class
    cdef size_t max_class
    cdef size_t s_id
    cdef size_t d_id
    cdef size_t l_start
    cdef size_t l_end
    cdef size_t r_start
    cdef size_t r_end


    cdef int transition(self, size_t clas, State *s) except -1
    cdef int* get_costs(self, State* s, size_t* heads, size_t* labels) except NULL
    cdef bint is_valid(self, size_t clas, size_t i, size_t n, size_t stack_len,
                      size_t head)
    cdef int* get_valid(self, State* s)
    cdef int break_tie(self, State* s, size_t* heads, size_t* labels) except -1
    cdef int s_cost(self, State *s, size_t* heads, size_t* labels)
    cdef int r_cost(self, State *s, size_t* heads, size_t* labels)
    cdef int d_cost(self, State *s, size_t* g_heads, size_t* g_labels)
    cdef int l_cost(self, State *s, size_t* heads, size_t* labels)

    
