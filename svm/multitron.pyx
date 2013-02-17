import sys
import math

from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cimport cython

cdef size_t MIN_UPD = 2

DEF MAX_PARAM = 15000000

cdef inline double get_weight(Feature* feat, uint64_t clas):
    if feat.seen[clas]:
        return feat.params[clas].w
    else:
        return 0


cdef inline void update_param(Feature* feat, uint64_t clas, uint64_t now, double weight):
    if feat.seen[clas]:
        param = feat.params[clas]
        param.acc += (now - param.last_upd) * param.w
        param.w += weight
        param.last_upd = now
    else:
        feat.seen[clas] = True
        feat.params[clas] = <Param*>malloc(sizeof(Param))
        feat.params[clas].w = weight
        feat.params[clas].acc = 0
        feat.params[clas].last_upd = now


cdef void free_feat(Feature* feat, size_t n):
    cdef size_t i
    for i in range(n):
        if feat.seen[i]:
            free(feat.params[i])
    free(feat.params)
    free(feat.seen)

cdef class MultitronParameters:
    """
    Labels and features must be non-negative integers with max values
    max_classes and max_params.
    The feature value 0 is ignored.
    """
    
    def __cinit__(self, max_classes, feat_thresh=0):
        cdef uint64_t i
        self.scores = <double *>malloc(max_classes * sizeof(double))
        self.W = <Feature**>malloc(MAX_PARAM * sizeof(Feature*))
        self.seen = <bint*>malloc(MAX_PARAM * sizeof(int64_t))
        for i in range(MAX_PARAM):
            self.seen[i] = False
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
        for f in range(self.n_params):
            if self.seen[f]:
                free_feat(self.W[f], self.true_nr_class)
        free(self.W)
        free(self.seen)

    cdef int64_t lookup_label(self, uint64_t label) except -1:
        assert label < self.max_classes
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
        feat.params = <Param**>malloc(self.true_nr_class * sizeof(Param*))
        feat.seen = <bint*>malloc(self.true_nr_class * sizeof(bint))
        for i in range(self.true_nr_class):
            feat.seen[i] = False
        feat.n_upd = 0
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
                free_feat(feat, self.true_nr_class)
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
        assert weight > 0
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                continue
            if f >= self.max_param or not self.seen[f]:
                self.add_feature(f)
            self.W[f].n_upd += 1
            update_param(self.W[f], pred_i, self.now, weight * -1)
            update_param(self.W[f], gold_i, self.now, weight)
       
    cdef double* get_scores(self, uint64_t n_feats, uint64_t* features):
        cdef uint64_t i, f, c
        cdef uint64_t n_classes = self.true_nr_class
        cdef double* scores = self.scores
        cdef uint64_t max_param = self.max_param
        cdef Feature* feat
        for c in range(n_classes):
            scores[c] = 0
        for i in range(n_feats):
            f = features[i]
            if f != 0 and f < max_param and self.seen[f]:
                feat = self.W[f]
                for c in range(n_classes):
                    scores[c] += get_weight(feat, c)
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
        cdef Feature* feat
        cdef Param* param
        # average
        for f in range(self.max_param):
            if not self.seen[f]:
                continue
            feat = self.W[f]
            for c in range(self.n_classes):
                if feat.seen[c]:
                    param = feat.params[c]
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
        for f in range(1, self.max_param):
            if not self.seen[f] or self.W[f].n_upd < MIN_UPD:
                continue
            feat = self.W[f]
            non_zeroes = []
            for i in range(self.true_nr_class):
                w = get_weight(feat, i)
                if w != 0:
                    non_zeroes.append('%d=%s ' % (self.labels[i], w))
            if non_zeroes:
                out.write(u'%d\t%s\n' % (f, ' '.join(non_zeroes)))
        out.close()
        #for f in range(1, self.max_param):
        #    if not self.seen[f] or self.W[f].n_upd < MIN_UPD:
        #        out.write(zeroes + u'\n')
        #        continue
        #    feat = self.W[f]
        #    for c in xrange(self.n_classes):
        #        if c < self.true_nr_class:
        #            out.write(u" %s" % get_weight(feat, c))
        #        else:
        #            out.write(u" 0")
        #    out.write(u"\n")
        #out.close()

    def load(self, in_):
        cdef Feature* feat
        cdef size_t i
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
            self.add_feature(f)
            feat = self.W[f]
            for param_str in weights.split():
                clas, w = param_str.split('=')
                clas = int(clas)
                i = self.label_to_i[clas]
                assert i != -1
                feat.seen[i] = True
                feat.params[i] = <Param*>malloc(sizeof(Param))
                feat.params[i].w = float(w)

            #weights = [float(w) for w in line.strip().split()]
            #if any(w != 0 for w in weights):
            #    unpruned += 1
            #    self.add_feature(f + 1)
            #    feat = self.W[f + 1]
            #    for clas, w in enumerate(weights):
            #        feat.seen[clas] = True
            #        feat.params[clas] = <Param*>malloc(sizeof(Param))
            #        feat.params[clas].w = w
            #    feat.n_upd = MIN_UPD + 1
        print 'Read %d/%d non-zero features' % (unpruned, n_feats)
