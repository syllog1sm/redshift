from libc.stdint cimport uint32_t
cimport svm.multitron

cdef extern from "linear.h":
    cdef struct feature_node:
        int index
        double value

    cdef cppclass problem:
        int l, n
        double *y
        feature_node **x
        double bias
        double *W # Instance weights

    cdef enum:
        L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, \
        L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL, L2R_L2LOSS_SVR = 11, \
        L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL

    cdef cppclass parameter:
        int solver_type
        double eps
        double C
        double p
        int nr_weight
        int *weight_label
        double* weight
        double p

    cdef cppclass model:
        parameter param
        int nr_class
        size_t nr_feature
        double *w
        int *label
        double bias
        unsigned long long max_idx

    cdef model* train(problem*, parameter*)

    cdef model* load_model(char*)

    cdef void save_model(char*, model*)

    cdef double predict_probability(model* model, feature_node*, double*)

    cdef double predict_values(model*, feature_node*, double*)

    cdef bint check_probability_model(model*)

cdef class Problem:
    cdef problem *thisptr
    cdef int i
    cdef int max_instances

    cdef add(self, int, double, feature_node*)
    cdef finish(self, unsigned long long, unsigned int)
    cpdef save(self, object)
    cdef void _malloc(self, int)
    cdef void _from_instances(self, object)
    cdef void _from_path(self, object)

cdef class Model:
    cdef int nr_class
    cdef int nr_feature
    cdef double *scores_array
    cdef int max_index
    cdef object path
    cdef bint is_trained
    cdef double C
    cdef double eps
    cdef int solver_type

    cdef int add_instance(self, int label, double weight, int n, size_t* feats) except -1
    cdef int predict_from_ints(self, int n, size_t* feats, bint* valid_classes) except -1
    cdef int predict_single(self, int n, size_t* feats) except -1
    
    # Python interface expected
    #def begin_adding_instances(self, int n_instances)
    #def save(self, path=None)
    #def load(self, path=None)
    #def train(self)

cdef class Perceptron(Model):
    cdef svm.multitron.MultitronParameters model

    cdef object pyize_feats(self, int n, size_t* feats)
    cdef int add_instance(self, int label, double weight, int n_feats, size_t* feats) except -1
    cdef int add_amb_instance(self, bint* valid_labels, double w, int n, size_t* feats) except -1
    cdef int predict_from_ints(self, int n_feats, size_t* feats, bint* valid_classes) except -1
    cdef int predict_single(self, int n, size_t* feat_array) except -1
    cdef float n_corr
    cdef float total


cdef class LibLinear(Model):
    cdef model *modelptr
    cdef parameter *paramptr
    cdef Problem problem
    # Model attributes
    cdef double *w
    cdef int *label
    cdef double bias
    cdef bint is_probability_model
    cdef int* labels_by_score

    cdef int add_instance(self, int label, double weight, int n, size_t*) except -1
    cdef int predict_from_features(self, feature_node*, bint* valid_classes) except -1
    cdef int predict_from_ints(self, int n, size_t*, bint* valid_classes) except -1
    cdef int predict_single(self, int n, size_t* feat_array) except -1

    cdef int _train(self, Problem p) except -1
