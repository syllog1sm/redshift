from redshift.sentence cimport Token
from redshift._state cimport SlotTokens

cdef int fill_context(size_t* context, SlotTokens* tokens) except -1
