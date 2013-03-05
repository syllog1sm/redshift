# cython: profile=True
import sys
import math

from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cimport cython

cdef size_t MIN_UPD = 2

DEF MAX_PARAM = 10000000

DEF MAX_DENSE = 500000

cdef void resize_feat(Feature* feat, size_t n):
    cdef size_t i
    feat.max_class = n
    cdef Param** new_params = <Param**>malloc(n * sizeof(Param*))
    for i in range(feat.n_class):
        new_params[i] = feat.params[i]
    free(feat.params)
    feat.params = new_params


cdef inline void update_param(Feature* feat, uint64_t clas, uint64_t now, double weight):
    cdef size_t i
    cdef Param** new_params
    i = feat.index[clas]
    if i != -1:
        param = feat.params[i]
        param.acc += (now - param.last_upd) * param.w
        param.w += weight
        param.last_upd = now
    else:
        # Resize vector if necessary
        if feat.n_class == feat.max_class:
            resize_feat(feat, feat.max_class * 2)
        i = feat.n_class
        feat.index[clas] = i
        feat.params[i] = <Param*>malloc(sizeof(Param))
        feat.params[i].w = weight
        feat.params[i].acc = 0
        feat.params[i].last_upd = now
        feat.params[i].clas = clas
        feat.n_class += 1

cdef inline void update_dense(size_t now, size_t nr_class, uint64_t f, uint64_t clas,
                              double weight, double* w, double* acc, size_t* last_upd):
    cdef uint64_t i = (f * nr_class) + clas
    acc[i] += (now - last_upd[i]) * w[i]
    w[i] += weight
    last_upd[i] = now


cdef void free_feat(Feature* feat):
    cdef size_t i
    for i in range(feat.n_class):
        if feat.index[i] != -1:
            free(feat.params[i])
    free(feat.params)
    free(feat.index)

cdef class MultitronParameters:
    """
    Labels and features must be non-negative integers with max values
    max_classes and max_params.
    The feature value 0 is ignored.
    """
    
    def __cinit__(self, max_classes, feat_thresh=0):
        cdef uint64_t i
        self.scores = <double *>malloc(max_classes * sizeof(double))
        self.max_dense = MAX_DENSE
        self.seen = <bint*>calloc(MAX_PARAM, sizeof(int64_t))
        self.w = <double*>calloc(MAX_DENSE * max_classes, sizeof(double))
        self.acc = <double*>calloc(MAX_DENSE * max_classes, sizeof(double))
        self.last_upd = <size_t*>calloc(MAX_DENSE, max_classes * sizeof(size_t))
        self.W = <Feature**>malloc(MAX_PARAM * sizeof(Feature*))
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
        for f in range(self.n_params):
            if self.seen[f]:
                free_feat(self.W[f])
        free(self.W)
        free(self.seen)
        free(self.w)
        free(self.acc)
        free(self.last_upd)

    cdef int64_t lookup_label(self, uint64_t label) except -1:
        assert label < self.max_classes, '%d must be < %d' % (label, self.max_classes)
        if self.label_to_i[label] >= 0:
            return self.label_to_i[label]
        else:
            self.label_to_i[label] = self.n_classes
            self.labels[self.n_classes] = label
            self.n_classes += 1
            return self.n_classes - 1

    cdef int64_t add_feature(self, uint64_t f) except -1:
        cdef uint64_t i
        assert f < MAX_PARAM
        if self.max_param <= f:
            self.max_param = f + 1
        cdef Feature* feat = <Feature*>malloc(sizeof(Feature))
        feat.params = <Param**>malloc(5 * sizeof(Param*))
        feat.index = <bint*>malloc(self.true_nr_class * sizeof(int))
        for i in range(self.true_nr_class):
            feat.index[i] = -1
        feat.n_upd = 0
        feat.n_class = 0
        feat.max_class = 5
        self.W[f] = feat
        self.seen[f] = True

    cdef int64_t prune_rares(self, size_t thresh) except -1:
        cdef uint64_t f
        cdef int64_t idx
        cdef uint64_t n_pruned = 0
        cdef Feature** W = self.W
        cdef Feature* feat
        cdef uint64_t new_max = 0
        self.n_params = 0
        for f in range(1, self.max_param):
            if not self.seen[f]:
                continue
            feat = W[f]
            if feat.n_upd < thresh:
                free_feat(feat)
                self.seen[f] = False
                n_pruned += 1
            else:
                self.n_params += 1
                if f > new_max:
                    new_max = f
        self.max_param = new_max
        print "Kept %d/%d" % (self.n_params, self.n_params + n_pruned)

    cdef tick(self):
        self.now = self.now + 1

    cdef int64_t update(self, uint64_t pred_label, uint64_t gold_label,
                    uint64_t n_feats, uint64_t* features, double weight) except -1:
        cdef size_t i
        cdef uint64_t f
        self.tick()
        cdef uint64_t gold_i = self.lookup_label(gold_label)
        cdef uint64_t pred_i = self.lookup_label(pred_label)
        if gold_i == pred_i:
            return 0
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                break
            self.update_single(gold_i, f, weight)
            self.update_single(pred_i, f, -weight)
    
    cdef int update_single(self, uint64_t label, uint64_t f, double weight) except -1:
        cdef uint64_t i = self.lookup_label(label)
        if f != 0 and weight != 0:
            if f < self.max_dense:
                update_dense(self.now, self.true_nr_class, f, i, weight,
                             self.w, self.acc, self.last_upd)
            else:
                if f >= self.max_param or not self.seen[f]:
                    self.add_feature(f)
                self.W[f].n_upd += 1
                update_param(self.W[f], i, self.now, weight)

    cdef inline int get_scores(self, uint64_t n_feats, uint64_t* features, double* scores) except -1:
        cdef uint64_t i, f, j, c
        cdef Param* param
        cdef uint64_t max_param = self.max_param
        cdef Feature* feat
        cdef uint64_t idx

        for c in range(self.true_nr_class):
            scores[c] = 0
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                break
            elif f < self.max_dense:
                idx = f * self.true_nr_class
                for c in range(self.true_nr_class):
                    scores[c] += self.w[idx + c]
            elif f < max_param and self.seen[f]:
                feat = self.W[f]
                for j in range(feat.n_class):
                    param = feat.params[j]
                    scores[param.clas] += param.w

    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features):
        cdef uint64_t i
        self.get_scores(n_feats, features, self.scores)
        cdef int best_i = 0
        cdef double best = self.scores[0]
        for i in range(self.n_classes):
            if best < self.scores[i]:
                best_i = i
                best = self.scores[i]
        return self.labels[best_i]

    cdef int64_t finalize(self) except -1:
        cdef uint64_t f
        cdef uint64_t c
        cdef Feature* feat
        cdef Param* param
        for f in range(self.max_dense * self.true_nr_class):
            self.acc[f] += (self.now - self.last_upd[f]) * self.w[f]
            self.w[f] = self.acc[f] / self.now
        # average
        for f in range(self.max_param):
            if not self.seen[f]:
                continue
            feat = self.W[f]
            for i in range(feat.n_class):
                param = feat.params[i]
                param.acc += (self.now - param.last_upd) * param.w
                param.w = param.acc / self.now

    def dump(self, out=sys.stdout):
        cdef uint64_t i
        cdef int64_t f, clas
        cdef Feature* feat
        # Write LibSVM compatible format
        out.write(u'solver_type L1R_LR\n')
        out.write(u'nr_class %d\n' % self.n_classes)
        out.write(u'label %s\n' % ' '.join([str(self.labels[i]) for i in
                                            range(self.n_classes)]))
        out.write(u'nr_feature %d\n' % self.max_param)
        out.write(u'bias -1\n')
        out.write(u'w\n')
        zeroes = '0 ' * self.n_classes
        # Break LibSVM compatibility for now to be a bit more disk-friendly
        for f in range(0, self.max_dense * self.true_nr_class, self.true_nr_class):
            non_zeroes = []
            for c in range(self.true_nr_class):
                if self.w[f + c] != 0:
                    non_zeroes.append(u'%d=%s' % (c, self.w[f + c]))
            out.write(u'%d\t%s\n' % (f / self.true_nr_class, u' '.join(non_zeroes)))
        for f in range(self.max_dense, self.max_param):
            if not self.seen[f] or self.W[f].n_upd < MIN_UPD:
                continue
            feat = self.W[f]
            non_zeroes = []
            for i in range(feat.n_class):
                param = feat.params[i]
                non_zeroes.append('%d=%s ' % (self.labels[param.clas], param.w))
            if non_zeroes:
                out.write(u'%d\t%s\n' % (f, ' '.join(non_zeroes)))
        out.close()

    def load(self, in_):
        cdef Feature* feat
        cdef size_t i
        print 'load'
        #cdef uint64_t f, clas, idx
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
        # Break LibSVM compatibility for now to be a bit more disk-friendly
        for line in in_:
            n_feats += 1
            unpruned += 1
            f, weights = line.split('\t')
            f = int(f)
            if f < self.max_dense:
                for param_str in weights.split():
                    raw_label, w = param_str.split('=')
                    i = self.label_to_i[int(raw_label)]
                    assert i != -1
                    self.w[(f * self.true_nr_class) + int(raw_label)] = float(w) 
                continue
            self.add_feature(f)
            feat = self.W[f]
            classes = []
            for param_str in weights.split():
                raw_label, w = param_str.split('=')
                i = self.label_to_i[int(raw_label)]
                assert i != -1
                classes.append((i, w))
            resize_feat(feat, len(classes))
            # It's slightly faster if we're accessing the scores array sequentially,
            # so sort the classes before we index them in our sparse array.
            for i, w in sorted(classes):
                feat.params[feat.n_class] = <Param*>malloc(sizeof(Param))
                feat.params[feat.n_class].w = float(w)
                feat.params[feat.n_class].clas = i
                feat.index[i] = feat.n_class
                feat.n_class += 1
        print 'done'
        print 'Read %d/%d non-zero features' % (unpruned, n_feats)
