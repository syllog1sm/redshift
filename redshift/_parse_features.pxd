from index.vocab cimport Word
from redshift.sentence cimport AnswerToken
from redshift._state cimport SlotTokens

cdef void fill_context(size_t* context, SlotTokens* tokens, AnswerToken* parse)
