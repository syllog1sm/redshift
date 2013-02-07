from libcpp.vector cimport vector

#cdef class MulticlassParamData:
#    cdef double *acc
#    cdef double *w
#    cdef int *lastUpd

cdef class MultitronParameters:
    cdef size_t n_classes
    cdef size_t n_params
    cdef size_t max_classes
    cdef size_t max_params
    cdef size_t now
    cdef double** acc
    cdef double** w
    cdef int** lastUpd
    cdef double* scores
    cdef list labels
    cdef dict label_to_i
    
    cdef tick(self)
    cdef int add_param(self, size_t f)
    cdef int add(self, size_t n_feats, size_t* features, int label, double amount) except -1
    cdef get_scores(self, size_t n_feats, size_t* features)
    cdef predict_best_class(self, size_t n_feats, size_t* features)
    cdef int _predict_best_class(self, size_t n_feats, size_t* features)
    cdef set seen_labels


    #cpdef pa_update(self, object gu_feats, object go_feats, int gu_cls, int go_cls,double C=?)
    #cpdef add_params(self, MultitronParameters other, double factor)
    #cpdef set(self, list features, int clas, double amount)
    #cpdef do_pa_update(self, list feats, int gold_cls, double C=?)
    #cpdef get_scores_r(self, features)
    #cpdef add_r(self, list features, int clas, double amount)
    #cpdef scalar_multiply(self, double scalar)
    #cdef _update(self, int goodClass, int badClass, list features)

    #cdef _update_r(self, int goodClass, int badClass, list features)

    #cpdef getW(self, object clas)

    #cpdef predict_best_class_r(self, list features)
