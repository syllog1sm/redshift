# cython: profile=True
import sys
import math

from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair

from cython.operator cimport dereference as deref, preincrement as inc

cimport cython

cdef void resize_feat(SparseFeature* feat, size_t n):
    cdef size_t i
    feat.max_class = n
    cdef Param** new_params = <Param**>malloc(n * sizeof(Param*))
    for i in range(feat.n_class):
        new_params[i] = feat.params[i]
    free(feat.params)
    feat.params = new_params


cdef inline void update_sparse_param(SparseFeature* feat, uint64_t clas, uint64_t now, double weight):
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


cdef inline void update_dense_param(size_t nr_class, size_t div, size_t now, size_t clas,
        double weight, DenseFeature* feat):
    cdef DenseParams* params
    cdef size_t part_idx = clas / div
    if not feat.seen[part_idx]:
        params = <DenseParams*>malloc(sizeof(DenseParams))
        params.w = <double*>calloc(div, sizeof(double))
        params.acc = <double*>calloc(div, sizeof(double))
        params.last_upd = <size_t*>calloc(div, sizeof(size_t))
        feat.parts[part_idx] = params[0]
        feat.seen[part_idx] = True
    else:
        params = &feat.parts[part_idx]
    cdef size_t i = clas % div
    params.acc[i] += (now - params.last_upd[i]) * params.w[i]
    params.w[i] += weight
    params.last_upd[i] = now


cdef void free_sparse_feat(SparseFeature* feat):
    cdef size_t i
    for i in range(feat.n_class):
        if feat.index[i] != -1:
            free(feat.params[i])
    free(feat.params)
    free(feat.index)


cdef void free_dense_feat(DenseFeature* feat, size_t div):
    cdef size_t i
    for i in range(div):
        if feat.seen[i]:
            free(feat.parts[i].w)
            free(feat.parts[i].acc)
            free(feat.parts[i].last_upd)
    free(feat.parts)
    free(feat.seen)
    free(feat)


cdef class MultitronParameters:
    def __cinit__(self, nr_class, feat_thresh=0):
        cdef uint64_t i
        self.scores = <double *>malloc(nr_class * sizeof(double))
        #self.max_dense = 200000
        #self.w = <DenseFeature*>malloc(self.max_dense * sizeof(DenseFeature))
        self.W = dense_hash_map[uint64_t, size_t]()
        self.W.set_empty_key(0)
        self.nr_class = nr_class
        self.div = <size_t>math.sqrt(nr_class) + 1
        self.now = 0

    def __dealloc__(self):
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        free(self.scores)
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            if data.second != 0:
                feat = <DenseFeature*>data.second
                free_dense_feat(feat, self.div)
            inc(it)


    cdef int _add_sparse_feature(self, uint64_t f) except -1:
        cdef uint64_t i
        cdef SparseFeature* feat = <SparseFeature*>malloc(sizeof(SparseFeature))
        feat.params = <Param**>malloc(5 * sizeof(Param*))
        feat.index = <bint*>malloc(self.nr_class * sizeof(int))
        for i in range(self.nr_class):
            feat.index[i] = -1
        feat.n_class = 0
        feat.max_class = 5
        self.W[f] = <size_t>feat

    cdef int add_feature(self, uint64_t f) except -1:
        cdef size_t i
        cdef DenseFeature* feat = <DenseFeature*>malloc(sizeof(DenseFeature))
        feat.parts = <DenseParams*>malloc(self.div * sizeof(DenseParams))
        feat.seen = <bint*>calloc(self.div, sizeof(bint))
        self.W[f] = <size_t>feat


    cdef int64_t prune_rares(self, size_t thresh) except -1:
        cdef uint64_t f
        cdef uint64_t n_kept = 0
        cdef uint64_t n_seen = 0
        cdef DenseFeature* feat
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        cdef size_t a

        self.W.set_deleted_key(1)
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            if data.second != 0:
                feat = <DenseFeature*>data.second
                a = 0
                for i in range(self.div):
                    if feat.seen[i]:
                        for j in range(self.div):
                            a += abs(<int>feat.parts[i].w[j])
                if a < thresh:
                    free_dense_feat(feat, self.div)
                    self.W.erase(it)
                else:
                    n_kept += 1
            n_seen += 1
            inc(it)
        self.W.clear_deleted_key()
        self.W.resize(n_kept * 2)
        print "%d/%d features kept" % (n_kept, n_seen)

    cdef tick(self):
        self.now = self.now + 1

    cdef int64_t update(self, size_t pred_i, size_t gold_i,
                    uint64_t n_feats, uint64_t* features, double weight) except -1:
        cdef size_t i
        cdef uint64_t f
        self.tick()
        if gold_i == pred_i:
            return 0
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                break
            self.update_single(gold_i, f, weight)
            self.update_single(pred_i, f, -weight)
    
    cdef int update_single(self, size_t cls, uint64_t f, double weight) except -1:
        cdef size_t feat_addr
        cdef DenseFeature* feat
        if f != 0 and weight != 0:
            feat_addr = self.W[f]
            if feat_addr == 0:
                self.add_feature(f)
            feat = <DenseFeature*>self.W[f]
            update_dense_param(self.nr_class, self.div, self.now, cls, weight, feat)

    cdef inline int get_scores(self, size_t n_feats, uint64_t* features, double* scores) except -1:
        cdef uint64_t i, f, j, k, c
        cdef size_t feat_addr
        cdef DenseFeature* feat
        cdef size_t part_idx
        cdef double* w

        for c in range(self.nr_class):
            scores[c] = 0
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                break
            feat_addr = self.W[f]
            if feat_addr != 0:
                feat = <DenseFeature*>feat_addr
                for j in range(self.div):
                    if feat.seen[j]:
                        part_idx = j * self.div
                        w = feat.parts[j].w
                        for k in range(self.div):
                            scores[part_idx + k] += w[k]

    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features):
        cdef uint64_t i
        self.get_scores(n_feats, features, self.scores)
        cdef int best_i = 0
        cdef double best = self.scores[0]
        for i in range(self.nr_class):
            if best < self.scores[i]:
                best_i = i
                best = self.scores[i]
        return best_i

    cdef int64_t finalize(self) except -1:
        cdef uint64_t f
        cdef DenseFeature* feat
        cdef DenseParams* params
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            if data.second != 0:
                feat = <DenseFeature*>data.second
                for i in range(self.div):
                    if feat.seen[i]:
                        params = &feat.parts[i]
                        for j in range(self.div):
                            params.acc[j] += (self.now - params.last_upd[j]) * params.w[j]
                            params.w[j] = params.acc[j] / self.now
            inc(it)

    def dump(self, out=sys.stdout):
        cdef uint64_t i
        cdef int64_t f, clas
        cdef DenseFeature* feat
        cdef DenseParams* params
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        # Write LibSVM compatible format
        out.write(u'nr_class %d\n' % (self.nr_class))
        zeroes = '0 ' * self.nr_class
        # Break LibSVM compatibility for now to be a bit more disk-friendly
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            if data.second != 0:
                feat = <DenseFeature*>data.second
                non_zeroes = []
                for i in range(self.div):
                    if feat.seen[i]:
                        params = &feat.parts[i]
                        for j in range(self.div):
                            if params.w[j]:
                                non_zeroes.append('%d=%s ' % ((i * self.div) + j,
                                                              params.w[j]))
                if non_zeroes:
                    out.write(u'%d\t%s\n' % (data.first, ' '.join(non_zeroes)))
            inc(it)
        out.close()

    def load(self, in_):
        cdef DenseFeature* feat
        cdef size_t i
        nr_feat = 0
        nr_weight = 0
        #cdef uint64_t f, clas, idx
        header = in_.readline()
        self.nr_class = int(header.split()[1])
        self.div = <int>math.sqrt(self.nr_class) + 1
        free(self.scores)
        self.scores = <double *>malloc(self.nr_class * sizeof(double))
        cdef uint64_t f
        print "Loading %d class..." % self.nr_class,
        for line in in_:
            f_str, weights_str = line.split('\t')
            f = int(f_str)
            if f == 0:
                continue
            self.add_feature(f)
            feat = <DenseFeature*>self.W[f]
            nr_feat += 1
            for param_str in weights_str.split():
                cls_str, w = param_str.split('=')
                i = int(cls_str)
                assert i < self.nr_class
                part_idx = i / self.div
                if not feat.seen[part_idx]:
                    params = <DenseParams*>malloc(sizeof(DenseParams))
                    params.w = <double*>calloc(self.div, sizeof(double))
                    params.acc = <double*>calloc(self.div, sizeof(double))
                    params.last_upd = <size_t*>calloc(self.div, sizeof(size_t))
                    feat.parts[part_idx] = params[0]
                    feat.seen[part_idx] = True
                idx = i % self.div
                feat.parts[part_idx].w[idx] = float(w)
                nr_weight += 1
        print "%d weights for %d features" % (nr_weight, nr_feat)
