from redshift.sentence cimport Sentence

from thinc.typedefs cimport atom_t


cdef int fill_context(atom_t* context, Sentence* sent, size_t ptag, size_t pptag,
                      size_t i)

