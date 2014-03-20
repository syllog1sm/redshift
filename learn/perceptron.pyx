# cython: profile=True
import math
import gzip
from libc.stdlib cimport *
from libcpp.vector cimport vector
from libcpp.utility cimport pair
from libcpp.queue cimport priority_queue
from libc.string cimport strtok, memset

from cython.operator cimport dereference as deref, preincrement as inc

cimport cython
cimport index.hashes


cdef DenseFeature* init_dense_feat(uint64_t feat_id, size_t nr_class):
    """A DenseFeature has a flat weight array, with length equal to the
    number of classes.  This is inefficient for rare features when many
    classes are used, as most class-weights will be zero.
    
    For more efficient estimation, we record the first and last non-zero class,
    so that when we do the dot product we can only iterate through that range.
    """
    cdef DenseFeature* feat = <DenseFeature*>malloc(sizeof(DenseFeature))
    feat.w = <double*>calloc(nr_class, sizeof(double))
    feat.acc = <double*>calloc(nr_class, sizeof(double))
    feat.last_upd = <size_t*>calloc(nr_class, sizeof(size_t))
    feat.id = feat_id
    feat.nr_seen = 0
    feat.s = 0
    feat.e = 0
    return feat


cdef size_t load_dense_feat(size_t nr_class, double* weights, size_t nr_seen,
                            DenseFeature* feat):
    cdef size_t nr_weight = 0
    cdef size_t clas
    feat.nr_seen = nr_seen
    for clas in range(nr_class):
        if weights[clas] != 0:
            nr_weight += 1
            feat.e = clas + 1
        feat.w[clas] = weights[clas]
        if nr_weight == 0:
            feat.s = clas
    return nr_weight


cdef void free_dense_feat(DenseFeature* feat):
    free(feat.w)
    free(feat.acc)
    free(feat.last_upd)
    free(feat)


cdef inline void score_dense_feat(double* scores, DenseFeature* feat):
    feat.nr_seen += 1
    cdef size_t c
    cdef double* w = feat.w
    for c in range(feat.s, feat.e):
        scores[c] += w[c]


cdef void update_dense(size_t now, double w, size_t clas, DenseFeature* raw):
    raw.acc[clas] += (now - raw.last_upd[clas]) * raw.w[clas]
    raw.w[clas] += w
    raw.last_upd[clas] = now
    if clas < raw.s:
        raw.s = clas
    if clas >= raw.e:
        raw.e = clas + 1


cdef inline SquareFeature* init_square_feat(uint64_t feat_id, size_t div):
    """A SquareFeature divides its parameters into a square 2d array of parameters,
    so that regions which are unoccupied do not need to be allocated or examined.
    This is a compromise between a sparse array and a dense array, and works best
    when the order of classes is meaningful --- i.e., when two classes near each
    other are more likely to have non-zero weights for a particular feature.

    When we do the dot product, we iterate through div, and check whether we've
    seen any non-zero weights along row i. If we have, we iterate through the
    row, and re-construct the class index by (i * div) + j. We then fetch
    feat.parts[i].w[j].
    """
    cdef SquareFeature* feat = <SquareFeature*>malloc(sizeof(SquareFeature))
    feat.nr_seen = 1
    feat.parts = <DenseParams*>malloc(div * sizeof(DenseParams))
    feat.seen = <bint*>calloc(div, sizeof(bint))
    return feat


cdef size_t load_square_feat(size_t nr_class, double* weights, size_t nr_seen,
                             size_t div, SquareFeature* feat):
    cdef DenseParams* params
    feat.nr_seen = nr_seen
    cdef size_t nr_weight = 0
    for clas in range(nr_class):
        part_idx = clas / div
        if not feat.seen[part_idx]:
            params = <DenseParams*>malloc(sizeof(DenseParams))
            params.w = <double*>calloc(div, sizeof(double))
            params.acc = <double*>calloc(div, sizeof(double))
            params.last_upd = <size_t*>calloc(div, sizeof(size_t))
            feat.parts[part_idx] = params[0]
            feat.seen[part_idx] = True
        idx = clas % div
        feat.parts[part_idx].w[idx] = weights[clas]
        nr_weight += 1
    return nr_weight


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


cdef inline void score_square_feat(double* scores, size_t div, size_t nr_class,
                                   SquareFeature* feat):
    cdef size_t j, k, part_idx
    feat.nr_seen  += 1
    for j in range(div):
        if feat.seen[j]:
            part_idx = j * div
            for k in range(div):
                if (part_idx + k) >= nr_class:
                    break
                scores[part_idx + k] += feat.parts[j].w[k]


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


cdef size_t MAX_ACTIVE = 1000


cdef class Perceptron:
    def __cinit__(self, max_classes, model_loc):
        self.path = model_loc
        self.nr_class = max_classes
        self.scores = <double *>calloc(max_classes, sizeof(double))
        self.W = dense_hash_map[uint64_t, size_t]()
        self.W.set_empty_key(0)
        self.div = <size_t>math.sqrt(max_classes)
        if (self.div * self.div) < max_classes:
            self.div += 1
        self.accept_new_feats = True
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
        self._active_dense = <DenseFeature**>calloc(MAX_ACTIVE, sizeof(size_t))
        self._active_square = <SquareFeature**>calloc(MAX_ACTIVE, sizeof(size_t))

    def __dealloc__(self):
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        free(self.scores)
        free(self._active_dense)
        free(self._active_square)
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

    def end_train_iter(self, iter_num, feat_thresh):
        pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
        acc = pc(self.n_corr, self.total)
        cache_use = pc(self.cache.n_hit, self.cache.n_hit + self.cache.n_miss + 1e-100)
        msg = "#%d: Moves %d/%d=%s" % (iter_num, self.n_corr, self.total, acc)
        if cache_use != 0:
            msg += '. Cache use %s' % cache_use
        print msg
        if iter_num % 2 == 1 and feat_thresh > 1:
            self.prune(feat_thresh)
        if iter_num < 3:
            self.reindex()
        self.n_corr = 0.0
        self.total = 0.0

    def set_classes(self, labels):
        self.nr_class = len(labels)
        self.div = <size_t>math.sqrt(self.nr_class) + 1
        for i in range(1, self.nr_raws):
            free(self.raws[i])
            self.raws[i] = init_dense_feat(0, self.nr_class)

    cdef int add_feature(self, uint64_t f) except -1:
        cdef size_t i
        cdef SquareFeature* feat = init_square_feat(f, self.div)
        self.W[f] = <size_t>feat

    cdef int add_instance(self, size_t label, double weight, int n, uint64_t* feats) except -1:
        """
        Add instance with 1 good label. Generalise to multi-label soon.
        """
        cdef int64_t pred = self.predict_best_class(feats)
        self.update(pred, label, feats, 1)
        return pred

    def batch_update(self, deltas):
        cdef size_t feat_addr
        self.now += 1
        for clas, feats in deltas.items():
            for f, d in feats.items():
                assert f != 0
                if d != 0:
                    feat_addr = self.W[f]
                    if feat_addr == 0:
                        if not self.accept_new_feats:
                            continue
                        self.add_feature(f)
                        feat_addr = self.W[f]
                    if feat_addr < self.nr_raws:
                        update_dense(self.now, d, clas, self.raws[feat_addr])
                    else:
                        update_square(self.nr_class, self.div,
                                      self.now, d, clas, <SquareFeature*>feat_addr)

    cdef int64_t update(self, size_t pred_i, size_t gold_i,
                    uint64_t* features, double margin) except -1:
        cdef size_t i
        cdef uint64_t f
        self.now += 1
        if gold_i == pred_i:
            return 0
        cdef size_t feat_addr
        cdef SquareFeature* feat
        i = 0
        while True:
            f = features[i]
            i += 1
            if f == 0:
                break
            feat_addr = self.W[f]
            if feat_addr == 0:
                if not self.accept_new_feats:
                    continue
                self.add_feature(f)
                feat_addr = self.W[f]
            if feat_addr < self.nr_raws:
                update_dense(self.now, 1.0, gold_i, self.raws[feat_addr])
                update_dense(self.now, -1.0, pred_i, self.raws[feat_addr])
            else:
                update_square(self.nr_class, self.div,
                              self.now, 1.0, gold_i, <SquareFeature*>feat_addr)
                update_square(self.nr_class, self.div,
                              self.now, -1.0, pred_i, <SquareFeature*>feat_addr)
   
    cdef inline int fill_scores(self, uint64_t* features, double* scores) except -1:
        cdef size_t feat_addr
        cdef size_t i
        cdef uint64_t f

        cdef SquareFeature** active_square = self._active_square
        cdef DenseFeature** active_dense = self._active_dense
        cdef DenseFeature** raws = self.raws
        cdef size_t nr_square = 0
        cdef size_t nr_dense = 0
        cdef size_t nr_raws = self.nr_raws
        # First collect the active features, in two lots --- dense and square
        for i in range(MAX_ACTIVE):
            f = features[i]
            if f == 0:
                break
            feat_addr = self.W[f]
            if feat_addr >= nr_raws:
                active_square[nr_square] = <SquareFeature*>feat_addr
                nr_square += 1
            elif feat_addr != 0:
                active_dense[nr_dense] = raws[feat_addr]
                nr_dense += 1
        # Now evaluate the features. Doing it this way improves cache locality,
        # giving a small efficiency improvement.
        cdef size_t nr_class = self.nr_class
        memset(scores, 0, nr_class * sizeof(double))
        for i in range(nr_dense):
            score_dense_feat(scores, active_dense[i])
        cdef size_t div = self.div
        for i in range(nr_square):
            score_square_feat(scores, div, nr_class, active_square[i])


    cdef uint64_t predict_best_class(self, uint64_t* features):
        cdef uint64_t i
        self.fill_scores(features, self.scores)
        cdef int best_i = 0
        cdef double best = self.scores[0]
        for i in range(self.nr_class):
            if best < self.scores[i]:
                best_i = i
                best = self.scores[i]
        return best_i

    cdef int64_t finalize(self) except -1:
        cdef uint64_t f
        cdef double tmp
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
                            # Save the unaveraged value in accs
                            tmp = params.w[j]
                            params.acc[j] += (self.now - params.last_upd[j]) * params.w[j]
                            params.w[j] = params.acc[j] / self.now
                            params.acc[j] = tmp
        for i in range(1, self.nr_raws):
            weights = self.raws[i].w
            accs = self.raws[i].acc
            last_upd = self.raws[i].last_upd
            for c in range(self.nr_class):
                # Save the unaveraged value in accs so that we can easily unaverage
                tmp = weights[c]
                accs[c] += (self.now - last_upd[c]) * weights[c]
                weights[c] = accs[c] / self.now
                accs[c] = tmp
    
    cdef int unfinalize(self) except -1:
        cdef double tmp
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
                            tmp = params.w[j] * self.now
                            params.w[j] = params.acc[j]
                            params.acc[j] = tmp
        for i in range(1, self.nr_raws):
            for c in range(self.nr_class):
                tmp = self.raws[i].w[c] * self.now
                self.raws[i].w[c] = self.raws[i].acc[c]
                self.raws[i].acc[c] = tmp

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
        out = gzip.open(str(out_loc), 'w')
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
                        non_zeroes.append('%d=%.3f' % (i, self.raws[feat_addr].w[i]))
            else:
                feat = <SquareFeature*>feat_addr
                for i in range(self.div):
                    if feat.seen[i]:
                        params = &feat.parts[i]
                        for j in range(self.div):
                            clas = (i * self.div) + j
                            if params.w[j]:
                                non_zeroes.append('%d=%.3f ' % (clas, params.w[j]))
                 
            if non_zeroes:
                out.write(u'%d\t%d\t%s\n' % (feat_id, nr_seen, ' '.join(non_zeroes)))
        out.close()

    def load(self, in_, size_t thresh=0):
        cdef SquareFeature* feat
        cdef size_t i
        nr_feat = 0
        nr_weight = 0
        #cdef uint64_t f, clas, idx
        in_ = gzip.open(in_, 'rb')
        header = in_.readline()
        self.nr_class = int(header.split()[1])
        self.div = <int>math.sqrt(self.nr_class)
        if (self.div * self.div) < self.nr_class:
            self.div += 1
        free(self.scores)
        self.scores = <double *>malloc(self.nr_class * sizeof(double))
        cdef uint64_t f
        cdef size_t nr_raws = 1
        print "Loading %d class..." % self.nr_class,
        cdef double* weights = <double*>calloc(self.nr_class, sizeof(double))
        cdef char* param_str
        cdef char* line
        cdef bytes py_line
        cdef double w
        cdef int cls
        cdef char* token
        for py_line in in_:
            line = <char*>py_line
            token = strtok(line, '\t')
            f = strtoull(token, NULL, 10)
            token = strtok(NULL, '\t')
            nr_seen = atoi(token)
            if f == 0:
                continue
            if nr_seen < thresh:
                continue
            for cls in range(self.nr_class):
                weights[cls] = 0
            token = strtok(NULL, '=')
            while token != NULL and token[0] != '\n':
                cls = atoi(token)
                token = strtok(NULL, ' ')
                w = atof(token)
                weights[cls] = w
                token = strtok(NULL, '=')
            if nr_raws < self.nr_raws:
                nr_weight += load_dense_feat(self.nr_class, weights, nr_seen,
                                             self.raws[nr_raws])
                self.raws[nr_raws].id = f
                self.W[f] = nr_raws
                nr_raws += 1
                nr_feat += 1
                continue
            else:
                self.add_feature(f)
                nr_weight += load_square_feat(self.nr_class, weights, nr_seen, self.div,
                                               <SquareFeature*>self.W[f])
                nr_feat += 1
        free(weights)
        print "%d weights for %d features" % (nr_weight, nr_feat)

    def flush_cache(self):
        self.cache.flush()

    def prune(self, size_t thresh):
        assert thresh > 1
        cdef dense_hash_map[uint64_t, size_t].iterator it = self.W.begin()
        cdef pair[uint64_t, size_t] data
        cdef uint64_t f_id
        cdef SquareFeature* feat
        cdef size_t n_pruned = 0
        cdef size_t n_feats = 0
        while it != self.W.end():
            data = deref(it)
            inc(it)
            f_id = data.first
            feat_addr = data.second
            if f_id == 0 or (feat_addr <= self.nr_raws):
                continue
            feat = <SquareFeature*>feat_addr
            if feat.nr_seen < thresh:
                free_square_feat(feat, self.div)
                self.W[f_id] = 0
                n_pruned += 1
            n_feats += 1
        self.W.clear_deleted_key()
        print "%d/%d pruned (f=%d)" % (n_pruned, n_feats, thresh)

    def reindex(self):
        """For efficiency, move the most frequent features to dense feature
        vectors, instead of the square representation"""
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
        cdef uint64_t feat_nr_seen
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
        if not q:
            return None
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
            if not vacancies:
                break
            assert f_id != 0
            f_addr = self.W[f_id]
            if f_addr > self.nr_raws:
                self._dense_to_raw(f_id, f_addr, vacancies.pop())
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
        self.W[f_id] = i
        assert self.W[f_id] < self.nr_raws
        assert raw.e <= self.nr_class, raw.e
