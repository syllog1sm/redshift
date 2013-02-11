# cython: profile=True
import sys
import math

from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair


cdef class MultitronParameters:
    """
    Labels and features must be non-negative integers with max values
    max_classes and max_params.
    The feature value 0 is ignored.
    """
    
    def __cinit__(self, max_classes, feat_thresh=0):
        cdef size_t i
        self.scores = <double *>malloc(max_classes * sizeof(double))
        #self.W = dense_hash_map[size_t, ParamData]()
        #self.W.set_empty_key(0)
        self.W = vector[ParamData]()
        self.weights = vector[vector[double]]()
        self.feat_idx = vector[int64_t]()
        self.max_param = 0
        self.n_params = 0
        self.max_classes = max_classes
        self.true_nr_class = max_classes
        self.n_classes = 0
        self.now = 0
        self.labels = <size_t*>malloc(max_classes * sizeof(size_t))
        self.label_to_i = <int*>malloc(max_classes * sizeof(int))
        for i in range(max_classes):
            self.label_to_i[i] = -1
            self.labels[i] = 0

    def __dealloc__(self):
        free(self.scores)
        free(self.labels)
        free(self.label_to_i)
        # TODO: Freeing
        #for f in range(self.n_params):
        #    p = &self.W[f]
        #    free(self.W[f].w)
        #    free(self.W[f].acc)
        #    free(self.W[f].lastUpd)

    cdef int lookup_label(self, size_t label) except -1:
        assert label < self.max_classes
        if self.label_to_i[label] >= 0:
            return self.label_to_i[label]
        else:
            self.label_to_i[label] = self.n_classes
            self.labels[self.n_classes] = label
            self.n_classes += 1
            return self.n_classes - 1

    cdef int add_param(self, size_t f) except -1:
        cdef size_t i
        cdef ParamData* p = <ParamData*>malloc(sizeof(ParamData))
        while self.max_param <= f:
            self.feat_idx.push_back(-1)
            self.max_param += 1
        self.feat_idx[f] = self.n_params
        p.acc = <double*>malloc(self.true_nr_class * sizeof(double))
        p.lastUpd = <int*>malloc(self.true_nr_class  * sizeof(int))
        w = new vector[double]()
        for i in range(self.true_nr_class):
            p.lastUpd[i] = 0
            p.acc[i] = 0
            w.push_back(0)
        self.W.push_back(p[0])
        self.weights.push_back(w[0])
        self.n_params += 1

    cdef tick(self):
        self.now = self.now + 1

    cdef int update(self, size_t pred_label, size_t gold_label,
                    size_t n_feats, size_t* features) except -1:
        cdef size_t t
        cdef size_t f
        cdef int64_t idx
        self.tick()
        cdef size_t gold_i = self.lookup_label(gold_label)
        cdef size_t pred_i = self.lookup_label(pred_label)
        if gold_i == pred_i:
            return 0
        cdef ParamData* p
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                continue
            if f > self.max_param or self.feat_idx[f] == -1:
                self.add_param(f)
            idx = self.feat_idx[f]
            p = &self.W[idx]
            p.acc[pred_i] += (self.now - p.lastUpd[pred_i]) * self.weights[idx][pred_i]
            p.acc[gold_i] += (self.now - p.lastUpd[gold_i]) * self.weights[idx][gold_i]
            self.weights[idx][pred_i] -= 1
            self.weights[idx][gold_i] += 1
            p.lastUpd[pred_i] = self.now
            p.lastUpd[gold_i] = self.now
        
    cdef double* get_scores(self, size_t n_feats, size_t* features):
        #cdef vector[double] weights
        cdef size_t i, f, c
        cdef size_t n_classes = self.n_classes
        cdef double* scores = self.scores
        for i in range(self.max_classes):
            scores[i] = 0
        for i in range(n_feats):
            f = features[i]
            if f < self.max_param:
                idx = self.feat_idx[f]
                if f != 0 and idx != -1:
                    for c in range(n_classes):
                        scores[c] = scores[c] + self.weights[idx][c]
        return scores

    cdef size_t predict_best_class(self, size_t n_feats, size_t* features):
        cdef size_t i
        cdef double* scores = self.get_scores(n_feats, features)
        cdef int best_i = 0
        cdef double best = scores[0]
        for i in range(self.n_classes):
            if best < scores[i]:
                best_i = i
                best = scores[i]
        return self.labels[best_i]

    cdef int finalize(self) except -1:
        cdef int64_t f
        cdef size_t c
        # average
        for f in range(self.n_params):
            for c in range(self.n_classes):
                self.W[f].acc[c] += (self.now - self.W[f].lastUpd[c]) * self.weights[f][c]
                self.weights[f][c] = self.W[f].acc[c] / self.now

    def dump(self, out=sys.stdout):
        cdef size_t c
        cdef int64_t f
        # Write LibSVM compatible format
        out.write(u'solver_type L1R_LR\n')
        out.write(u'nr_class %d\n' % self.n_classes)
        out.write(u'label %s\n' % ' '.join([str(self.labels[i]) for i in
                                            range(self.n_classes)]))
        out.write(u'nr_feature %d\n' % self.max_param)
        out.write(u'bias -1\n')
        out.write(u'w\n')
        zeroes = '0 ' * self.n_classes
        for f in range(1, self.max_param):
            f = self.feat_idx[f]
            if f == -1:
                out.write(zeroes + u'\n')
                continue
            for c in xrange(self.n_classes):
                out.write(u" %s" % self.weights[f][c])
            out.write(u"\n")
        out.close()

    def load(self, in_=sys.stdin):
        cdef size_t f, clas
        header, data = in_.read().split('w\n')
        for line in header.split('\n'):
            if line.startswith('label'):
                label_names = line.strip().split()
                # Remove the word "label"
                label_names.pop(0)
        for label in label_names:
            self.lookup_label(int(label))
        self.true_nr_class = len(label_names)
        for f, line in enumerate(data.strip().split('\n')):
            weights = [float(w) for w in line.strip().split()]
            assert len(weights) == len(label_names)
            if any([w != 0 for w in weights]):
                self.add_param(f + 1)
                idx = self.feat_idx[f + 1]
                for clas, w in enumerate(weights):
                    self.weights[idx][clas] = w
        print 'loaded'
