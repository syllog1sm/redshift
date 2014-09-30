from redshift.sentence cimport Sentence


cdef int fill_context(size_t* context, Sentence* sent, size_t ptag, size_t pptag,
                      size_t i)

