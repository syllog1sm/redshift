#include <yepCore.h>
#include <yepMath.h>
#include <yepLibrary.h>


int add_inplace(double* x, const double* y, int length);


int add_inplace(double* x, const double* y, int length) {
    enum YepStatus status;
    status = yepCore_Add_IV64fV64f_IV64f(x, y, length);
    return status;
}

int add_into(double* out, const double* x, const double* y, int length) {
    enum YepStatus status;
    status = yepCore_Add_V64fV64f_V64f(x, y, out, length);
    return status;
}

int init_yeppp() {
    enum YepStatus status;
    /* Initialize the Yeppp! library */
    status = yepLibrary_Init();
    assert(status == YepStatusOk);
    return status;
}
