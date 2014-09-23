from cymem.cymem cimport Pool
from ._state cimport State
from .sentence cimport Input
from .sentence cimport Token
from .ae_transitions cimport Transition


cdef class PyState:
    cdef Pool mem
    cdef Input sent
    cdef State* state
    cdef list left_labels
    cdef list right_labels
    cdef list encoded_left
    cdef list encoded_right
    cdef size_t nr_moves
    cdef Transition* moves
    cdef dict moves_by_name
    cdef Token* gold
