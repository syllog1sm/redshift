# cython: profile=True
import math
from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.queue cimport priority_queue

from cython.operator cimport dereference as deref, preincrement as inc

cimport cython
cimport index.hashes


cdef DenseFeature* init_dense_feat(uint64_t feat_id, size_t nr_class):
    cdef DenseFeature* feat = <DenseFeature*>malloc(sizeof(DenseFeature))
    feat.w = <double*>calloc(nr_class, sizeof(double))
    feat.acc = <double*>calloc(nr_class, sizeof(double))
    feat.last_upd = <size_t*>calloc(nr_class, sizeof(size_t))
    feat.id = feat_id
    feat.nr_seen = 0
    feat.s = 0
    feat.e = 0
    return feat


cdef void free_dense_feat(DenseFeature* feat):
    free(feat.w)
    free(feat.acc)
    free(feat.last_upd)
    free(feat)


cdef void update_dense(size_t now, double w, size_t clas, DenseFeature* raw):
    raw.acc[clas] += (now - raw.last_upd[clas]) * raw.w[clas]
    raw.w[clas] += w
    raw.last_upd[clas] = now
    if clas < raw.s:
        raw.s = clas
    if clas >= raw.e:
        raw.e = clas + 1


cdef inline void update_square(size_t nr_class, size_t div,
                               size_t now, double weight, size_t clas, SquareFeature* feat):
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


cdef void free_square_feat(SquareFeature* feat, size_t div):
    cdef size_t i
    for i in range(div):
        if feat.seen[i]:
            free(feat.parts[i].w)
            free(feat.parts[i].acc)
            free(feat.parts[i].last_upd)
    free(feat.parts)
    free(feat.seen)
    free(feat)


cdef class Perceptron:
    # From Model
    def __cinit__(self, max_classes, model_loc, clean=False):
        self.path = model_loc
        self.nr_class = max_classes
        self.scores = <double *>calloc(max_classes, sizeof(double))
        self.W = dense_hash_map[uint64_t, size_t]()
        self.W.set_empty_key(0)
        self.div = <size_t>math.sqrt(max_classes) + 1
        self.now = 0
        self.nr_raws = 10000
        self.raws = <DenseFeature**>malloc(self.nr_raws * sizeof(DenseFeature*))
        cdef size_t i
        for i in range(self.nr_raws):
            self.raws[i] = init_dense_feat(0, max_classes)
        self.n_corr = 0.0
        self.total = 0.0
        self.use_cache = True
        self.cache = index.hashes.ScoresCache(max_classes) 

    def __dealloc__(self):
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        free(self.scores)
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            inc(it)
            if data.second >= self.nr_raws:
                free_square_feat(<SquareFeature*>data.second, self.div)
        cdef size_t i
        for i in range(self.nr_raws):
            free_dense_feat(self.raws[i])
        free(self.raws)

    def set_classes(self, labels):
        self.nr_class = len(labels)
        self.div = <size_t>math.sqrt(self.nr_class) + 1
        for i in range(1, self.nr_raws):
            free(self.raws[i])
            self.raws[i] = init_dense_feat(0, self.nr_class)

    cdef int add_feature(self, uint64_t f) except -1:
        cdef size_t i
        cdef SquareFeature* feat = <SquareFeature*>malloc(sizeof(SquareFeature))
        addr = <size_t>feat
        assert addr > self.nr_raws, addr
        feat.nr_seen = 1
        feat.parts = <DenseParams*>malloc(self.div * sizeof(DenseParams))
        feat.seen = <bint*>calloc(self.div, sizeof(bint))
        self.W[f] = <size_t>feat

    cdef int add_instance(self, size_t label, double weight, int n, uint64_t* feats) except -1:
        """
        Add instance with 1 good label. Generalise to multi-label soon.
        """
        cdef int64_t pred = self.predict_best_class(n, feats)
        self.update(pred, label, n, feats, 1)
        return pred

    def batch_update(self, deltas, margin):
        cdef size_t feat_addr
        cdef SquareFeature* feat
        self.now += 1
        for clas, feats in deltas.items():
            for f, d in feats.items():
                assert f != 0
                if d != 0:
                    feat_addr = self.W[f]
                    if feat_addr == 0:
                        self.add_feature(f)
                    elif feat_addr < self.nr_raws:
                        update_dense(self.now, d, clas, self.raws[feat_addr])
                    else:
                        update_square(self.nr_class, self.div,
                                      self.now, d, clas, <SquareFeature*>feat_addr)

    cdef int64_t update(self, size_t pred_i, size_t gold_i,
                    uint64_t n_feats, uint64_t* features, double margin) except -1:
        cdef size_t i
        cdef uint64_t f
        self.now += 1
        if gold_i == pred_i:
            return 0
        weight = 1.0
        cdef size_t feat_addr
        cdef SquareFeature* feat
        for i in range(n_feats):
            f = features[i]
            if f == 0:
                break
            if weight == 0:
                continue
            feat_addr = self.W[f]
            if feat_addr == 0:
                self.add_feature(f)
            elif feat_addr < self.nr_raws:
                update_dense(self.now, 1.0, gold_i, self.raws[feat_addr])
                update_dense(self.now, -1.0, pred_i, self.raws[feat_addr])
            else:
                update_square(self.nr_class, self.div,
                              self.now, 1.0, gold_i, <SquareFeature*>feat_addr)
                update_square(self.nr_class, self.div,
                              self.now, -1.0, pred_i, <SquareFeature*>feat_addr)
   
    cdef inline int fill_scores(self, size_t n, uint64_t* features, double* scores) except -1:
        cdef size_t i, f, j, k, c
        cdef size_t feat_addr
        cdef SquareFeature* feat
        cdef DenseFeature* raw_feat
        cdef size_t part_idx
        for c in range(self.nr_class):
            scores[c] = 0
        i = 0
        while True:
            f = features[i]
            if f == 0:
                break
            i += 1
            feat_addr = self.W[f]
            if feat_addr == 0:
                continue
            elif feat_addr < self.nr_raws:
                raw_feat = self.raws[feat_addr]
                raw_feat.nr_seen += 1
                for c in range(raw_feat.s, raw_feat.e):
                    scores[c] += raw_feat.w[c]
            else:
                feat = <SquareFeature*>feat_addr
                feat.nr_seen += 1
                for j in range(self.div - 1):
                    if feat.seen[j]:
                        part_idx = j * self.div
                        for k in range(self.div):
                            scores[part_idx + k] += feat.parts[j].w[k]
                j = self.div - 1
                if feat.seen[j]:
                    part_idx = j * self.div
                    for k in range(self.nr_class - part_idx):
                        scores[part_idx + k] += feat.parts[j].w[k]

    cdef uint64_t predict_best_class(self, uint64_t n_feats, uint64_t* features):
        cdef uint64_t i
        self.fill_scores(n_feats, features, self.scores)
        cdef int best_i = 0
        cdef double best = self.scores[0]
        for i in range(self.nr_class):
            if best < self.scores[i]:
                best_i = i
                best = self.scores[i]
        return best_i

    cdef int64_t finalize(self) except -1:
        cdef uint64_t f
        cdef SquareFeature* feat
        cdef DenseParams* params
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            inc(it)
            if data.second >= self.nr_raws:
                feat = <SquareFeature*>data.second
                for i in range(self.div):
                    if feat.seen[i]:
                        params = &feat.parts[i]
                        for j in range(self.div):
                            params.acc[j] += (self.now - params.last_upd[j]) * params.w[j]
                            params.w[j] = params.acc[j] / self.now
        for i in range(1, self.nr_raws):
            weights = self.raws[i].w
            accs = self.raws[i].acc
            last_upd = self.raws[i].last_upd
            for c in range(self.nr_class):
                accs[c] += (self.now - last_upd[c]) * weights[c]
                weights[c] = accs[c] / self.now

    def save(self, out_loc):
        cdef size_t i
        cdef uint64_t feat_id
        # Break LibSVM compatibility for now to be a bit more disk-friendly
        by_nr_seen = []
        for i in range(1, self.nr_raws):
            by_nr_seen.append((self.raws[i].nr_seen, self.raws[i].id))
        cdef dense_hash_map[uint64_t, size_t].iterator it
        cdef pair[uint64_t, size_t] data
        cdef SquareFeature* feat
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            inc(it)
            feat_addr = data.second
            if feat_addr >= self.nr_raws:
                feat = <SquareFeature*>feat_addr
                by_nr_seen.append((feat.nr_seen, data.first))
        by_nr_seen.sort(reverse=True)
        out = open(str(out_loc), 'w')
        # Write LibSVM compatible format
        out.write(u'nr_class %d\n' % (self.nr_class))
        zeroes = '0 ' * self.nr_class 
        cdef DenseParams* params
        for nr_seen, feat_id in by_nr_seen:
            feat_addr = self.W[feat_id]
            non_zeroes = []
            if feat_addr == 0:
                continue
            elif feat_addr < self.nr_raws:
                for i in range(self.nr_class):
                    if self.raws[feat_addr].w[i] != 0:
                        non_zeroes.append('%d=%s' % (i, self.raws[feat_addr].w[i]))
            else:
                feat = <SquareFeature*>feat_addr
                for i in range(self.div):
                    if feat.seen[i]:
                        params = &feat.parts[i]
                        for j in range(self.div):
                            clas = (i * self.div) + j
                            if params.w[j]:
                                non_zeroes.append('%d=%s ' % (clas, params.w[j]))
                 
            if non_zeroes:
                out.write(u'%d\t%d\t%s\n' % (feat_id, nr_seen, ' '.join(non_zeroes)))
        out.close()

    def load(self, in_):
        cdef SquareFeature* feat
        cdef size_t i
        nr_feat = 0
        nr_weight = 0
        #cdef uint64_t f, clas, idx
        in_ = open(in_)
        header = in_.readline()
        self.nr_class = int(header.split()[1])
        self.div = <int>math.sqrt(self.nr_class) + 1
        free(self.scores)
        self.scores = <double *>malloc(self.nr_class * sizeof(double))
        cdef uint64_t f
        cdef size_t nr_raws = 1
        print "Loading %d class..." % self.nr_class,
        for line in in_:
            f_str, nr_seen, weights_str = line.split('\t')
            f = int(f_str)
            if f == 0:
                continue
            if int(nr_seen) < 100:
                continue
            if nr_raws < self.nr_raws:
                seen_cls = False
                for param_str in weights_str.split():
                    nr_weight += 1
                    cls_str, w = param_str.split('=')
                    self.raws[nr_raws].w[int(cls_str)] = float(w)
                    if not seen_cls:
                        self.raws[nr_raws].s = int(cls_str)
                        seen_cls = True
                self.raws[nr_raws].e = int(cls_str) + 1
                self.raws[nr_raws].id = f
                self.raws[nr_raws].nr_seen = int(nr_seen)
                self.W[f] = nr_raws
                nr_raws += 1
                nr_feat += 1
                continue
            self.add_feature(f)
            feat = <SquareFeature*>self.W[f]
            nr_feat += 1
            for param_str in weights_str.split():
                cls_str, w = param_str.split('=')
                i = int(cls_str)
                assert i < self.nr_class
                part_idx = i / self.div
                feat.nr_seen = int(nr_seen)
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

    def flush_cache(self):
        self.cache.flush()

    def prune(self, thresh):
        pass

    def reindex(self):
        def get_nr_seen(size_t feat_addr):
            cdef SquareFeature* feat
            if feat_addr < self.nr_raws:
                return self.raws[feat_addr].nr_seen
            else:
                feat = <SquareFeature*>feat_addr
                return feat.nr_seen
        cdef dense_hash_map[uint64_t, size_t].iterator it = self.W.begin()
        cdef pair[uint64_t, size_t] data
        # Build priority queue of the top N scores
        cdef uint64_t f_id
        cdef int feat_nr_seen
        q = []
        while it != self.W.end():
            data = deref(it)
            inc(it)
            f_id = data.first
            feat_addr = data.second
            if f_id == 0 or feat_addr == 0:
                continue
            feat_nr_seen = get_nr_seen(feat_addr)
            q.append((feat_nr_seen, f_id))
        q.sort(reverse=True)
        cutoff = q[self.nr_raws][0]
        vacancies = []
        for i in range(1, self.nr_raws):
            if self.raws[i].id == 0:
                vacancies.append(i)
            elif self.raws[i].nr_seen < cutoff:
                assert self.raws[i].id != 0
                self._raw_to_dense(self.raws[i].id, i)
                vacancies.append(i)
        for freq, f_id in q[:self.nr_raws]:
            assert f_id != 0
            f_addr = self.W[f_id]
            if f_addr > self.nr_raws:
                self._dense_to_raw(f_id, f_addr, vacancies.pop())
                if not vacancies:
                    break
        assert not vacancies, str(vacancies)

    def _raw_to_dense(self, uint64_t f_id, size_t raw_idx):
        self.add_feature(f_id)
        feat = <SquareFeature*>self.W[f_id]
        feat.nr_seen = self.raws[raw_idx].nr_seen
        self.raws[raw_idx].id = 0
        self.raws[raw_idx].nr_seen = 0
        assert self.raws[raw_idx].s <= self.nr_class
        cdef double* weights = self.raws[raw_idx].w
        cdef double* accs = self.raws[raw_idx].acc
        cdef size_t* last_upd = self.raws[raw_idx].last_upd
        cdef size_t i, clas, part_idx
        cdef DenseParams* params
        for clas in range(self.nr_class):
            if weights[clas] == 0:
                continue
            part_idx = clas / self.div
            if not feat.seen[part_idx]:
                params = <DenseParams*>malloc(sizeof(DenseParams))
                params.w = <double*>calloc(self.div, sizeof(double))
                params.acc = <double*>calloc(self.div, sizeof(double))
                params.last_upd = <size_t*>calloc(self.div, sizeof(size_t))
                feat.parts[part_idx] = params[0]
                feat.seen[part_idx] = True
            else:
                params = &feat.parts[part_idx]
            i = clas % self.div
            params.acc[i] = accs[clas]
            params.w[i] = weights[clas]
            params.last_upd[i] = last_upd[i]

    def _dense_to_raw(self, uint64_t f_id, size_t feat_addr, size_t i):
        cdef double* w
        cdef double* accs
        cdef size_t* last_upd
        cdef DenseFeature* raw = self.raws[i]
        raw.id = f_id
        cdef SquareFeature* feat = <SquareFeature*>feat_addr
        raw.nr_seen = feat.nr_seen
        raw.s = 0
        raw.e = 0
        cdef size_t j, k
        for j in range(self.div):
            if feat.seen[j]:
                part_idx = j * self.div
                w = feat.parts[j].w
                accs = feat.parts[j].acc
                last_upd = feat.parts[j].last_upd
                for k in range(self.div):
                    clas = part_idx + k
                    if clas >= self.nr_class:
                        break
                    if w[k] != 0:
                        if raw.s == 0:
                            raw.s = clas
                        raw.e = clas + 1
                    raw.w[clas] = w[k]
                    raw.acc[clas] = accs[k]
                    raw.last_upd[clas] = last_upd[k]

        free_square_feat(feat, self.div)
        self.W.erase(f_id)
        self.W[f_id] = i
        assert self.W[f_id] < self.nr_raws
        assert raw.e <= self.nr_class, raw.e
