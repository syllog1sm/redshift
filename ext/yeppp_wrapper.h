#include <yepCore.h>
#include <yepMath.h>
#include <yepLibrary.h>


int add_inplace(double* x, const double* y, int length);


int add_inplace(double* x, const double* y, int length) {
    enum YepStatus status;
    status = yepCore_Add_IV64fV64f_IV64f(x, y, length);
    return status;
}


