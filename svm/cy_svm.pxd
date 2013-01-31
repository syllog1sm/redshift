from libc.stdint cimport uint32_t
from libcpp.utility cimport pair
from libcpp.vector cimport vector


ctypedef double Label

cdef extern from "sparsehash/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        size_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin()
        iterator end()
        size_t size()
        size_t max_size()
        bint empty()
        size_t bucket_count()
        size_t bucket_size(size_t i)
        size_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(size_t n)
        void rehash(size_t n)
        dense_hash_map()
        dense_hash_map(size_t n)


        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        size_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil



cdef extern from *:
    ctypedef void* const_void "const void*"


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

    cdef add(self, Label, double, feature_node*)
    cdef finish(self, unsigned long long, unsigned int)
    cpdef save(self, object)
    cdef void _malloc(self, int)
    cdef void _from_instances(self, object)
    cdef void _from_path(self, object)

cdef class Model:
    cdef model *modelptr
    cdef parameter *paramptr
    cdef Problem problem
    # Model attributes
    cdef int nr_class
    cdef int nr_feature
    cdef double *w
    cdef int *label
    cdef double bias
    cdef double *scores_array
    cdef bint is_probability_model
    cdef int max_index
    cdef object path
    cdef bint is_trained
    cdef int* labels_by_score
    cdef double C

    cdef int add_instance(self, Label, double weight, int n, size_t*) except -1
    cdef double predict_from_features(self, feature_node*, bint* valid_classes) except -1
    cdef double predict_from_ints(self, int n, size_t*, bint* valid_classes) except -1
    cdef double predict_single(self, int n, size_t* feat_array) except -1

    cdef int _train(self, Problem p) except -1
