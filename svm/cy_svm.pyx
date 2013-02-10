# cython: profile=True
from libc.stdlib cimport malloc, free
import os
import sys
import numpy as np
from numpy cimport uint64_t

from pathlib import Path

cimport numpy as np
cimport cython.operator


cdef extern from *:
    ctypedef void* const_void "const void*"


ctypedef np.int_t DTYPE_t

cdef class Problem:
    def __cinit__(self, *non_kw_args, instances=None, path=None, length=None):
        self.thisptr = new problem()
        self.i = 0
        kwargs = [a for a in [instances, path, length] if a is not None]
        self.max_instances = 0
        if non_kw_args or len(kwargs) != 1:
            raise TypeError("Problem takes exactly 1 named argument "
                            "(instances, path or length)")
        if length is not None:
           self._malloc(<int>length)
        elif path is not None:
            if not isinstance(path, Path):
                path = Path(path)
            self._from_path(path)
        elif instances is not None:
            self._from_instances(instances)
        

    cdef add(self, int label, double weight, feature_node* feat_seq):
        cdef size_t i = 0
        while feat_seq[i].index != -1:
            assert feat_seq[i].index < 100000000
            i += 1
        self.thisptr.y[self.i] = label
        self.thisptr.x[self.i] = feat_seq
        self.thisptr.W[self.i] = weight
        assert self.i < self.max_instances, self.max_instances
        self.i += 1

    cdef finish(self, unsigned long long max_index, unsigned int n_instances):
        cdef size_t i, j
        cdef feature_node *x
        cdef size_t new_idx
        cdef size_t n_kept = 0
        cdef size_t total = 0
        self.thisptr.l = n_instances
        self.thisptr.bias = -1
        self.thisptr.n = max_index
        classes = set()
        for i in range(n_instances):
            classes.add(self.thisptr.y[i])
        n_classes = len(classes)
        print("%d classes, %d instances, %d features" % (n_classes, n_instances, max_index))

    cdef void _malloc(self, int n):
        self.thisptr.x = <feature_node **>malloc((n) * sizeof(feature_node*))
        self.thisptr.y = <double *>malloc(n * sizeof(double))
        self.thisptr.W = <double *>malloc(n * sizeof(double))
        self.max_instances = n

    cdef void _from_instances(self, object instances):
        # Ignore efficiency in preparing the data, in favour of accepting pure
        # python iterables for the instances object, and being more memory safe
        cdef int j
        cdef size_t max_index = 0
        cdef feature_node* c_feats
        cdef int n_instances = len(instances)
        
        self._malloc(n_instances)
        
        for weight, label, feats in instances:
            c_feats = seq_to_features(feats)
            self.add(weight, label, c_feats)
            for j in range(len(feats)):
                if c_feats[j].index > max_index:
                    max_index = c_feats[j].index
        self.finish(max_index, n_instances)
        
    cdef void _from_path(self, object train_path):
        cdef int max_index = 0
        cdef feature_node* c_feats
        lines = list(train_path.open().read().strip().split('\n'))
        cdef int n_instances = len(lines)
        
        self._malloc(n_instances)
        
        for line in lines:
            pieces = line.split()
            label = int(pieces.pop(0))
            feats = []
            for feat_str in pieces:
                index, value = feat_str.split(':')
                index = int(index)
                feats.append((index, float(value)))
                if index > max_index:
                    max_index = index
            c_feats = seq_to_features(feats)
            self.add(label, 1.0, c_feats)        
        self.finish(max_index, n_instances)

    cpdef save(self, path):
        cdef feature_node* x
        cdef int i, j
        cdef double y
        cdef double w
        cdef size_t idx
        path = Path(path)
        out = path.open('w')
        for i in range(<int>self.thisptr.l):
            x = self.thisptr.x[i]
            j = 0
            feats = set()
            while x[j].index != -1:
                idx = x[j].index
                # LibSVM needs features to be unique
                feats.add(idx)
                j += 1
            y = self.thisptr.y[i]
            w = self.thisptr.W[i]
            feat_line = u' '.join([u'%d:1' % f for f in sorted(feats)])
            out.write(u'%.2f %d %s\n' % (w, y, feat_line))
        out.close()

    def __dealloc__(self):
        for i in range(self.thisptr.l):
            free(self.thisptr.x[i])
        free(self.thisptr.x)
        free(self.thisptr.y)
        free(self.thisptr.W)
        free(self.thisptr)

    # For testing
    def get_feature(self, int i, int j):
        if i >= self.i:
            raise IndexError
        else:
            return self.thisptr.x[i][j].index

    def get_label(self, int i):
        if i >= self.i:
            raise IndexError
        else:
            return self.thisptr.y[i]


TRAIN_IN_PROCESS = True


cdef class Model:
    """Base class for learners"""
    cdef int add_instance(self, int label, double weight, int n, size_t* feat_indices) except -1:
        return -1

    cdef int predict_from_ints(self, int n, size_t* feat_array, bint* valid_classes) except -1:
        return -1

    cdef int predict_single(self, int n, size_t* feat_array) except -1:
        return -1

    cdef int update(self, size_t pred, size_t gold, int n, size_t* feats) except -1:
        return -1

    def begin_adding_instances(self, n_instances):
        raise NotImplemented

    def save(self, path):
        raise NotImplemented

    def load(self, path):
        raise NotImplemented

# If this is a problem we can get the client to specify it
MAX_FEATS = 10000000
cdef class Perceptron(Model):
    def __cinit__(self, max_classes, model_loc, int solver_type=14, float C=1,
                  float eps=0.01, clean=False):
        self.path = model_loc
        self.nr_class = max_classes
        self.model = svm.multitron.MultitronParameters(max_classes)
        # C is the smoothing parameter for LibLinear, and eps is the tolerance
        # If we need these hyper-parameters in perceptron sometime, here they are
        self.C = C
        self.eps = eps
        self.solver_type = solver_type
        self.n_corr = 0.0
        self.total = 0.0

    def set_nr_class(self, nr_class):
        self.nr_class = nr_class
        self.model.true_nr_class = nr_class

    def begin_adding_instances(self, size_t n_feats):
        pass

    cdef int add_instance(self, int label, double w, int n, size_t* feats) except -1:
        """
        Add instance with 1 good label. Generalise to multi-label soon.
        """
        cdef size_t pred = self.model.predict_best_class(n, feats)
        self.update(pred, label, n, feats)
        return pred

    cdef int update(self, size_t pred, size_t gold, int n, size_t* feats) except -1:
        self.model.update(pred, gold, n, feats)
        if gold == pred:
            self.n_corr += 1
        self.total += 1

    def train(self):
        self.model.finalize()

    cdef int predict_from_ints(self, int n, size_t* feats, bint* valid_classes) except -1:
        cdef:
            size_t i
            size_t best_class
            size_t label
            double score, best_score
            size_t* labels
        cdef bint seen_valid = False
        cdef double* scores = self.model.get_scores(n, feats)
        labels = self.model.labels
        best_score = 0
        best_class = 0
        for i in range(self.model.n_classes):
            score = scores[i]
            label = labels[i]
            if valid_classes[label] and (score > best_score or not seen_valid):
                best_score = score
                best_class = label
                seen_valid = True
        # If can't find a valid label, add a previously unseen valid label
        if not seen_valid:
            for i in range(self.model.max_classes):
                if valid_classes[i]:
                    self.model.lookup_label(i)
                    return i
            else:
                raise StandardError
        return best_class


    cdef int predict_single(self, int n, size_t* feats) except -1:
        return self.model.predict_best_class(n, feats)

    def save(self, model_loc):
        self.model.dump(model_loc.open('w'))

    def load(self, model_loc):
        self.model.load(model_loc.open())


cdef class LibLinear(Model):
    def __cinit__(self, n_classes, model_loc, int solver_type=L2R_L2LOSS_SVC_DUAL,
                  float C=1, float eps=-1, clean=False):
        self.paramptr = new parameter()
        self.paramptr.solver_type = solver_type
        self.paramptr.C = C
        if eps == -1:
            if solver_type in (0, 2):
                eps = 0.01
            elif solver_type in (5, 6, 11):
                eps = 0.001
            elif solver_type in (1, 3, 4, 7, 12, 13):
                eps = 0.1
            else:
                raise StandardError
        self.paramptr.eps = eps
        self.p = 0.1
        self.paramptr.nr_weight = 0
        self.path = Path(model_loc)
        if self.path.exists():
            if clean:
                print "Removing existing model"
                self.path.unlink()
            else:
                self.load(self.path)
        self.is_trained = False
    
    def begin_adding_instances(self, n_instances):
        """Set up instance-by-instance data addition, so that only the
        memory-efficient C++ array needs to be in memory, not the list of
        Python objects."""
        self.problem = Problem(length=n_instances)
        self.max_index = 0

    cdef int add_instance(self, int label, double weight, int n, size_t* feat_indices) except -1:
        x = ints_to_features(n, feat_indices)
        self.problem.add(label, weight, x)
        for i in range(n):
            if feat_indices[i] > self.max_index:
                self.max_index = feat_indices[i]

    def train(self, *non_kw_args, instances=None, path=None):
        if non_kw_args or len([a for a in [instances, path] if a is not None]) > 1:
            raise TypeError("Model.train takes at most 1 named argument "
                            "(instances or path)")
        if instances is not None:
            self.problem = Problem(instances=instances)
        elif path is not None:
            self.problem = Problem(path=path)
        elif not self.problem:
            raise TypeError("Model.train requires 1 named argument unless data "
                            "has already been added (instances or path)")
        else:
            self.problem.finish(self.max_index, self.problem.i) 
        if not TRAIN_IN_PROCESS:
            self.problem.save(self.path.parent().join('train.libsvm'))
            self.is_trained = False
        else:
            self._train(self.problem)
            self.is_trained = True

    cdef int _train(self, Problem p) except -1:
        # TODO: Where's this get set to zero?
        #self.paramptr.C = self.C
        self.modelptr = train(<problem *>p.thisptr, self.paramptr)
        self._setup()

    cdef int predict_from_ints(self, int n, size_t* feat_array, bint* valid_classes) except -1:
        cdef feature_node* features = ints_to_features(n, feat_array)
        value = self.predict_from_features(features, valid_classes)
        free(features)
        return value

    cdef int predict_single(self, int n, size_t* feat_array) except -1:
        cdef feature_node* features = ints_to_features(n, feat_array)
        scores_array = <double*>malloc(self.nr_class * sizeof(double))
        value = <int>predict_values(self.modelptr, features, self.scores_array)
        free(scores_array)
        free(features)
        return value

    cdef int predict_from_features(self, feature_node* features, bint* valid_classes) except -1:
        cdef size_t i
        cdef int value, label
        cdef double score
        value = <int>predict_values(self.modelptr, features, self.scores_array)
        if valid_classes[<int>value]:
            return value
        cdef int best_label = 0
        cdef double best_score = -1000000.0
        for i in range(self.nr_class):
            label = self.modelptr.label[i]
            if not valid_classes[<int>label]:
                continue
            score = self.scores_array[i]
            if score > best_score:
                best_score = score
                best_label = label
        assert best_score > -1000000.0
        return best_label

    def save(self, path=None):
        if path is None:
            path = self.path

        path = str(path)
        save_model(path, self.modelptr)

    def load(self, path=None):
        if path is None:
            path = self.path
        assert not Path(path).is_dir()
        path = str(path)
        cdef model* m = load_model(path)
        self.modelptr = m
        self._setup()

    def _setup(self):
        cdef int i
        self.is_trained = True
        self.scores_array = <double *>malloc(self.modelptr.nr_class * sizeof(double))
        self.is_probability_model = check_probability_model(self.modelptr)
        self.nr_class = self.modelptr.nr_class
        self.nr_feature = self.modelptr.nr_feature
        self.w = self.modelptr.w
        self.label = self.modelptr.label
        self.bias = self.modelptr.bias
    
    def __dealloc__(self):
        free(self.scores_array)
        free(self.modelptr)
        free(self.paramptr)

    property scores:
        def __get__(self):
            cdef np.npy_intp dims = 1
            py_array = np.empty(self.nr_class, dtype=np.double)
            for i in range(self.nr_class):
                py_array[i] = self.scores_array[i]
            return py_array

    property labels:
        def __get__(self):
            py_list = []
            for i in range(self.nr_class):
                py_list.append(self.modelptr.label[i])
            return py_list

    property nr_class:
        def __get__(self):
            return self.nr_class

    # Parameter handling
    property solver_type:
        def __get__(self): return self.paramptr.solver_type
        def __set__(self, int value): self.paramptr.solver_type = value
 
    property C:
        def __get__(self): return self.paramptr.C
        def __set__(self, int value): self.paramptr.C = value
 
    property eps:
        def __get__(self): return self.paramptr.eps
        def __set__(self, double value): self.paramptr.eps = value

    property p:
        def __get__(self): return self.paramptr.p
        def __set__(self, double value):
            self.paramptr.p = value

    property path:
        def __get__(self): return self.path

cdef feature_node* err

cdef feature_node * seq_to_features(object instance) except *:
    cdef feature_node feature, term_feat
    cdef feature_node* x
    cdef int i
    instance.sort()
    x = <feature_node *>malloc((len(instance) + 1) * sizeof(feature_node))
    for i, feat_idx in enumerate(instance):
        if isinstance(feat_idx, int):
            x[i] = feature_node(index=feat_idx, value=1.0)
        else:
            x[i] = feature_node(index=feat_idx[0], value=feat_idx[1])
    x[i + 1] = feature_node(index=-1, value=1.0)
    return x

cdef feature_node* ints_to_features(int n, size_t* ints) except NULL:
    cdef int value
    cdef int i, j
    cdef int non_zeroes = 0
    for i in range(n):
        if ints[i] != 0:
            non_zeroes += 1
    cdef feature_node* features = <feature_node*>malloc((non_zeroes + 1) * sizeof(feature_node))
    j = 0
    for i in range(n):
        if ints[i] != 0:
            features[j] = feature_node(index=ints[i], value=1.0)
            j += 1
    features[j] = feature_node(index=-1, value=1.0)
    return features
