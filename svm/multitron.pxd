cdef class MulticlassParamData:
    cdef double *acc
    cdef double *w
    cdef int *lastUpd

cdef class MultitronParameters:
    cdef int nclasses
    cdef int now
    cdef dict W

    cdef double* scores

    cpdef getW(self, object clas)

    cdef _tick(self)

    cpdef scalar_multiply(self, double scalar)

    cpdef add(self, list features, int clas, double amount)

    cpdef add_r(self, list features, int clas, double amount)

    cpdef set(self, list features, int clas, double amount)

    cpdef add_params(self, MultitronParameters other, double factor)

    cpdef do_pa_update(self, list feats, int gold_cls, double C=?)

    cpdef pa_update(self, object gu_feats, object go_feats, int gu_cls, int go_cls,double C=?)
    
    cpdef get_scores(self, object features)

    cpdef get_scores_r(self, features)
   
    cpdef predict_best_class_r(self, list features)
     
    cpdef predict_best_class(self, list features)

    cdef int _predict_best_class(self, list features)

    cdef _update(self, int goodClass, int badClass, list features)

    cdef _update_r(self, int goodClass, int badClass, list features)
