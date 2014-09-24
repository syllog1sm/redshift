from cymem.cymem cimport Pool
from ._state cimport State
from .sentence cimport Input
from .sentence cimport Token


include "compile_time_options.pxi"
IF TRANSITION_SYSTEM == 'arc_eager':
    from .arc_eager cimport *
ELSE:
    from .arc_hybrid cimport *

cdef class PyState:
    cdef Pool mem
    cdef Input sent
    cdef State* state
    cdef list left_labels
    cdef list right_labels
    cdef list dfl_labels
    cdef list encoded_left
    cdef list encoded_right
    cdef list encoded_dfl
    cdef size_t nr_moves
    cdef Transition* moves
    cdef dict moves_by_name
    cdef Token* gold
