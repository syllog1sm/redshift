cdef extern from "yeppp_wrapper.h":
    int add_inplace(double* x, const double* y, size_t length)
