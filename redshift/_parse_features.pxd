from redshift.sentence cimport Token
from redshift._state cimport SlotTokens
from thinc.typedefs cimport atom_t, weight_t

cdef int fill_context(atom_t* context, SlotTokens* tokens, Token* parse) except -1
