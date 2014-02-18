from _state cimport Kernel, Subtree
from sentence cimport Sentence


#cdef void fill_context(size_t* context, size_t nr_label, size_t* words,
#                       size_t* tags, int* pauses,
#                       size_t* clusters, size_t* cprefix6s, size_t* cprefix4s#,
#                       size_t* orths, int* parens, int* quotes,
#                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l)
# 


cdef void fill_context(size_t* context, size_t nr_label, Sentence* sent, Kernel* k)
