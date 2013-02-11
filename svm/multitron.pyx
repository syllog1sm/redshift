# cython: profile=True
import sys
import math

from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair

PRUNE_THRESH = 1e-10

cdef class MultitronParameters:
    """
    Labels and features must be non-negative integers with max values
    max_classes and max_params.
    The feature value 0 is ignored.
    """
    
    def __cinit__(self, max_classes, feat_thresh=0):
        cdef size_t i
        self.scores = <double *>malloc(max_classes * sizeof(double))
        self.W = dense_hash_map[size_t, ParamData]()
        self.param_freqs = dense_hash_map[size_t, size_t]()
        self.W.set_empty_key(0)
        self.param_freqs.set_empty_key(0)
        self.feat_thresh = 0
        self.count_freqs = False
        self.max_param = 0
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
        cdef ParamData* p
        if f > self.max_param:
            self.max_param = f
        self.param_freqs[f] = 1
        p = <ParamData*>malloc(sizeof(ParamData))
        p.acc = <double*>malloc(self.true_nr_class * sizeof(double))
        p.w = <double*>malloc(self.true_nr_class * sizeof(double))
        p.lastUpd = <int*>malloc(self.true_nr_class  * sizeof(int))
        for i in range(self.true_nr_class):
            p.lastUpd[i] = 0
            p.acc[i] = 0
            p.w[i] = 0
        self.W[f] = p[0]

    cdef tick(self):
        self.now = self.now + 1

    cdef int update(self, size_t pred_label, size_t gold_label,
                    size_t n_feats, size_t* features) except -1:
        cdef size_t i, f
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
            if self.param_freqs[f] == 0:
                self.add_param(f)
            elif self.count_freqs:
                self.param_freqs[f] += 1
                continue
            if self.param_freqs[f] < self.feat_thresh:
                continue
            p = &self.W[f]
            p.acc[pred_i] += (self.now - p.lastUpd[pred_i]) * p.w[pred_i]
            p.acc[gold_i] += (self.now - p.lastUpd[gold_i]) * p.w[gold_i]
            p.w[pred_i] -= 1
            p.w[gold_i] += 1
            p.lastUpd[pred_i] = self.now
            p.lastUpd[gold_i] = self.now
        
    cdef double* get_scores(self, size_t n_feats, size_t* features):
        cdef size_t i, f, c
        cdef double* w
        cdef double* scores = self.scores
        for i in range(self.max_classes):
            scores[i] = 0
        cdef size_t n_classes = self.n_classes
        for i in range(n_feats):
            f = features[i]
            if f != 0 and self.param_freqs[f] > self.feat_thresh:
                w = self.W[f].w
                for c in range(n_classes):
                    scores[c] += w[c]
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
        cdef size_t f, c
        # average
        for f in range(1, self.max_param):
            if self.param_freqs[f] == 0:
                continue
            for c in range(self.n_classes):
                self.W[f].acc[c] += (self.now - self.W[f].lastUpd[c]) * self.W[f].w[c]
                self.W[f].w[c] = self.W[f].acc[c] / self.now

    def dump(self, out=sys.stdout):
        cdef size_t f, c
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
            if self.param_freqs[f] == 0:
                out.write(zeroes + u'\n')
                continue
            for c in xrange(self.n_classes):
                out.write(u" %s" % self.W[f].w[c])
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
        n_feats = 0
        n_unpruned = 0
        for f, line in enumerate(data.strip().split('\n')):
            weights = [float(w) for w in line.strip().split()]
            assert len(weights) == len(label_names)
            n_feats += 1
            if any([(abs(w) >= PRUNE_THRESH and w != 0) for w in weights]):
                self.add_param(f + 1)
                for clas, w in enumerate(weights):
                    self.W[f + 1].w[clas] = w
                n_unpruned += 1
        print "%d/%d feats >= %f" % (n_unpruned, n_feats, PRUNE_THRESH)
