from libc.stdlib cimport calloc, free


cdef class Pool:
    def __cinit__(self):
        self._addresses = set()

    def __dealloc__(self):
        cdef size_t addr
        for addr in self._addresses:
            free(<void*>addr)

    cdef void* safe_alloc(self, size_t number, size_t size) except NULL:
        cdef void* addr = calloc(number, size)
        assert <size_t>addr not in self._addresses
        self._addresses.add(<size_t>addr)
        return addr


cdef class Memory:
    def __cinit__(self, size_t number, size_t size):
        self.addr = <size_t>calloc(number, size)

    def __dealloc__(self):
        free(<void*>self.addr)
