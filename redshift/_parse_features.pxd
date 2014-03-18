from index.vocab cimport Word
from redshift.sentence cimport AnswerToken, Step
from redshift._state cimport SlotTokens

cdef int fill_context(size_t* context, SlotTokens* tokens, AnswerToken* parse,
                      Step* lattice) except -1
