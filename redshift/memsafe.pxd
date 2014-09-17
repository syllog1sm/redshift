cdef class Pool:
    cdef set _addresses

    cdef void* safe_alloc(self, size_t number, size_t size) except NULL


cdef class Memory:
    cdef size_t addr
