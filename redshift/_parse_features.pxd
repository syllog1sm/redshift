from _state cimport Kernel, Subtree


cdef void fill_context(size_t* context, size_t nr_label, size_t* words,
                       size_t* tags,
                       size_t* clusters, size_t* cprefix6s, size_t* cprefix4s,
                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l)
 
