import sys
import math

from libc.stdlib cimport *

cdef class MultitronParameters:
    def __cinit__(self, max_classes, max_params): 
        self.scores = <double *>malloc(max_classes * sizeof(double))
        self.w = <double**>malloc(max_params * sizeof(double*))
        self.acc = <double**>malloc(max_params * sizeof(double*))
        self.lastUpd = <int**>malloc(max_params * sizeof(double*))
        self.max_classes = max_classes
        self.max_params = max_params
        self.n_params = 0
        self.n_classes = 0
        self.now = 0
        self.labels = []
        self.label_to_i = {}

    def __dealloc__(self):
        free(self.scores)
        for f in range(self.n_params):
            free(self.w[f])
            free(self.acc[f])
            free(self.lastUpd[f])
        free(self.w)
        free(self.acc)
        free(self.lastUpd)

    def lookup_label(self, label):
        if label in self.label_to_i:
            return self.label_to_i[label]
        else:
            self.label_to_i[label] = self.n_classes
            self.labels.append(label)
            self.n_classes += 1
            assert self.n_classes < self.max_classes
            return self.n_classes - 1

    cdef int add_param(self, size_t f) except -1:
        assert f < self.max_params
        while f >= self.n_params:
            self.w[self.n_params] = <double*>malloc(self.max_classes * sizeof(double))
            self.acc[self.n_params] = <double*>malloc(self.max_classes * sizeof(double))
            self.lastUpd[self.n_params] = <int*>malloc(self.max_classes * sizeof(int))
            for i in range(self.max_classes):
                self.lastUpd[self.n_params][i] = 0
                self.acc[self.n_params][i] = 0
                self.w[self.n_params][i] = 0
            self.n_params += 1

    cdef tick(self):
        self.now = self.now + 1

    cdef int add(self, size_t n_feats, size_t* features, int label, double amount) except -1:
        cdef size_t i, f
        cdef size_t clas = self.lookup_label(label)
        cdef double** acc = self.acc
        cdef double** w = self.w
        cdef int** lastUpd = self.lastUpd
        for i in range(n_feats):
            f = features[i]
            if f >= self.n_params:
                self.add_param(f)
            acc[f][clas] += (self.now - lastUpd[f][clas]) * w[f][clas]
            w[f][clas] += amount
            lastUpd[f][clas] = self.now
        
    cdef get_scores(self, size_t n_feats, size_t* features):
        cdef size_t i, f, c
        cdef double* scores = self.scores
        for i in range(self.max_classes):
            scores[i] = 0
        cdef double** w = self.w
        cdef size_t n_classes = self.n_classes
        for i in range(n_feats):
            f = features[i]
            if f < self.n_params:
                for c in range(n_classes):
                    scores[c] += w[f][c]

    cdef predict_best_class(self, size_t n_feats, size_t* features):
        best_i = self._predict_best_class(n_feats, features)
        return self.labels[best_i]

    cdef int _predict_best_class(self, size_t n_feats, size_t* features):
        cdef size_t i
        self.get_scores(n_feats, features)
        cdef int best_i = 0
        cdef double best = self.scores[0]
        for i in range(self.n_classes):
            if best < self.scores[i]:
                best_i = i
                best = self.scores[i]
        return best_i

    def finalize(self):
        cdef size_t f, c
        # average
        for f in range(self.n_params):
            for c in range(self.n_classes):
                self.acc[f][c] += (self.now - self.lastUpd[f][c]) * self.w[f][c]
                self.w[f][c] = self.acc[f][c] / self.now

    def dump(self, out=sys.stdout):
        cdef size_t f, c
        # Write LibSVM compatible format
        out.write(u'solver_type L1R_LR\n')
        out.write(u'nr_class %d\n' % self.n_classes)
        out.write(u'label %s\n' % ' '.join(map(str, self.labels)))
        out.write(u'nr_feature %d\n' % self.n_params)
        out.write(u'bias -1\n')
        out.write(u'w\n')
        for f in range(1, self.n_params):
            for c in xrange(self.n_classes):
                out.write(u" %s" % self.w[f][c])
            out.write(u"\n")
        out.close()

    def load(self, in_=sys.stdin):
        header, data = in_.read().split('w\n')
        for line in header.split('\n'):
            if line.startswith('label'):
                label_names = line.strip().split()
                # Remove the word "label"
                label_names.pop(0)
        for label in label_names:
            self.lookup_label(int(label))
        for i, line in enumerate(data.strip().split('\n')):
            pieces = line.strip().split()
            assert len(pieces) == len(label_names)
            self.add_param(i + 1)
            for j, w in enumerate(pieces):
                self.w[i + 1][j] = float(w)
