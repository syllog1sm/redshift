from _state cimport Kernel, Subtree
from sentence cimport Sentence


cdef void fill_context(size_t* context, size_t nr_label, Sentence* sent, Kernel* k)
