# cython: profile=True
import sys
import math

from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef size_t MIN_UPD = 5

cdef class MultitronParameters:
    """
    Labels and features must be non-negative integers with max values
    max_classes and max_params.
    The feature value 0 is ignored.
    """
    
    def __cinit__(self, max_classes, feat_thresh=0):
        cdef uint64_t i
        self.scores = <double *>malloc(max_classes * sizeof(double))
        #self.W = dense_hash_map[uint64_t, ParamData]()
        #self.W.set_empty_key(0)
        self.W = vector[ParamData]()
        self.feat_idx = vector[int64_t]()
        self.max_param = 0
        self.n_params = 0
        self.max_classes = max_classes
        self.true_nr_class = max_classes
        self.n_classes = 0
        self.now = 0
        self.labels = <uint64_t*>malloc(max_classes * sizeof(uint64_t))
        self.label_to_i = <int64_t*>malloc(max_classes * sizeof(int64_t))
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

    cdef int64_t lookup_label(self, uint64_t label) except -1:
        assert label < self.max_classes
        if self.label_to_i[label] >= 0:
            return self.label_to_i[label]
        else:
            self.label_to_i[label] = self.n_classes
            self.labels[self.n_classes] = label
            self.n_classes += 1
            return self.n_classes - 1

    cdef int64_t add_param(self, uint64_t f) except -1:
        cdef uint64_t i
        while self.max_param <= (f + 1):
            self.feat_idx.push_back(-1)
            self.max_param += 1
        cdef ParamData* p = <ParamData*>malloc(sizeof(ParamData))
        p.w = <double*>malloc(self.true_nr_class * sizeof(double))
        p.acc = <double*>malloc(self.true_nr_class * sizeof(double))
        p.last_upd = <size_t*>malloc(self.true_nr_class * sizeof(size_t))
        for i in range(self.true_nr_class):
            p.w[i] = 0
            p.acc[i] = 0
            p.last_upd[i] = 0
        p.n_upd = 0
        self.feat_idx[f] = self.n_params
        self.n_params += 1
        self.W.push_back(p[0])

    cdef tick(self):
        self.now = self.now + 1

    cdef int64_t update(self, uint64_t pred_label, uint64_t gold_label,
                    uint64_t n_feats, uint64_t* features) except -1:
        cdef double* w
        cdef double* acc
        cdef size_t* last_upd
        cdef uint64_t f
        cdef int64_t idx
        self.tick()
        cdef uint64_t gold_i = self.lookup_label(gold_label)
        cdef uint64_t pred_i = self.lookup_label(pred_label)
        if gold_i == pred_i:
            return 0
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                continue
            if f > self.max_param or self.feat_idx[f] == -1:
                self.add_param(f)
            idx = self.feat_idx[f]
            self.W[idx].n_upd += 1
            w = self.W[idx].w
            acc = self.W[idx].acc
            last_upd = self.W[idx].last_upd
            acc[pred_i] += (self.now - last_upd[pred_i]) * w[pred_i]
            acc[gold_i] += (self.now - last_upd[gold_i]) * w[gold_i]
            w[pred_i] -= 1
            w[gold_i] += 1
            last_upd[pred_i] = self.now
            last_upd[gold_i] = self.now
        
    cdef double* get_scores(self, uint64_t n_feats, uint64_t* features):
        cdef uint64_t i, f, c
        cdef uint64_t n_classes = self.n_classes
        cdef double* scores = self.scores
        cdef int64_t idx
        cdef uint64_t max_param = self.max_param
        cdef double* w
        cdef double score
        for c in range(n_classes):
            scores[c] = 0
        for i in range(n_feats):
            f = features[i]
            if f != 0 and f < max_param:
                idx = self.feat_idx[f]
                if idx != -1:
                    score = 0
                    w = self.W[idx].w
                    for c in range(n_classes):
                        scores[c] += w[c]
        return scores

    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features):
        cdef uint64_t i
        cdef double* scores = self.get_scores(n_feats, features)
        cdef int best_i = 0
        cdef double best = scores[0]
        for i in range(self.n_classes):
            if best < scores[i]:
                best_i = i
                best = scores[i]
        return self.labels[best_i]

    cdef int64_t finalize(self) except -1:
        cdef uint64_t f
        cdef uint64_t c
        # average
        for f in range(self.n_params):
            for c in range(self.n_classes):
                self.W[f].acc[c] += (self.now - self.W[f].last_upd[c]) * self.W[f].w[c]
                self.W[f].w[c] = self.W[f].acc[c] / self.now

    def dump(self, out=sys.stdout):
        cdef uint64_t c
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
            if f == -1 or self.W[f].n_upd < MIN_UPD:
                out.write(zeroes + u'\n')
                continue
            for c in xrange(self.n_classes):
                out.write(u" %s" % self.W[f].w[c])
            out.write(u"\n")
        out.close()

    def load(self, in_=sys.stdin):
        cdef uint64_t f, clas, idx
        for line in in_:
            if line.startswith('label'):
                label_names = line.strip().split()
                # Remove the word "label"
                label_names.pop(0)
            if line == 'w\n':
                break
        for label in label_names:
            self.lookup_label(int(label))
        self.true_nr_class = len(label_names)
        n_feats = 0
        unpruned = 0
        for f, line in enumerate(in_):
            n_feats += 1
            weights = [float(w) for w in line.strip().split()]
            if any(w != 0 for w in weights):
                unpruned += 1
                self.add_param(f + 1)
                idx = self.feat_idx[f + 1]
                for clas, w in enumerate(weights):
                    self.W[idx].w[clas] = w
                self.W[idx].n_upd = MIN_UPD + 1
        print 'Read %d/%d non-zero features' % (unpruned, n_feats)
