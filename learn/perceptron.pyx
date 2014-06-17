# cython: profile=True
import os.path
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

DEF LINE = 8

cdef size_t get_div(size_t nr_class):
    return (nr_class / LINE) + 1

cdef double score_dense_feat(double* scores, DenseFeature* feat) nogil:
    feat.nr_seen += 1
    cdef size_t c
    cdef double* w = feat.w
    cdef double t = 0
    for c in range(feat.s, feat.e):
        scores[c] += w[c]
        t += w[c]
    return t


cdef double score_square_feat(double* scores, size_t nr_class, SquareFeature* feat):
    cdef size_t i, j, part_idx, clas
    feat.nr_seen  += 1
    cdef double t = 0
    cdef size_t div = get_div(nr_class)
    cdef double* weights
    cdef bint* seen = feat.seen
    for i in range(div):
        if seen[i]:
            weights = feat.parts[i].w
            part_idx = i * LINE
            for j in range(LINE if nr_class - part_idx >= LINE else nr_class - part_idx):
                scores[part_idx + j] += weights[j]
                t += weights[j]
    return t


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


cdef size_t load_dense_feat(size_t nr_class, double unary_weight, double* weights,
                            size_t nr_seen, DenseFeature* feat):
    cdef size_t nr_weight = 0
    cdef size_t clas
    feat.nr_seen = nr_seen
    for clas in range(nr_class):
        if weights[clas] != 0:
            feat.e = clas + 1
            if nr_weight == 0:
                feat.s = clas
            nr_weight += 1
        feat.w[clas] = weights[clas]
    return nr_weight


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


cdef inline SquareFeature* init_square_feat(uint64_t feat_id, size_t nr_class):
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
    cdef size_t div = get_div(nr_class)
    cdef SquareFeature* feat = <SquareFeature*>malloc(sizeof(SquareFeature))
    feat.nr_seen = 1
    feat.parts = <DenseParams**>calloc(div, sizeof(DenseParams*))
    feat.seen = <bint*>calloc(div, sizeof(bint))
    return feat


cdef size_t load_square_feat(size_t nr_class, double unary_weight, double* weights,
                             size_t nr_seen, SquareFeature* feat):
    feat.nr_seen = nr_seen
    cdef size_t nr_weight = 0
    cdef size_t part
    cdef size_t clas
    cdef size_t div = get_div(nr_class)
    for part in range(div):
        for i in range(LINE):
            clas = (part * LINE) + i
            if clas >= nr_class:
                break
            if weights[clas] != 0:
                if not feat.seen[part]:
                    feat.parts[part] = <DenseParams*>calloc(1, sizeof(DenseParams))
                    feat.seen[part] = True
                feat.parts[part].w[i] = weights[clas]
                nr_weight += 1
    return nr_weight


cdef void free_square_feat(SquareFeature* feat, size_t nr_class):
    cdef size_t div = get_div(nr_class)
    cdef size_t i
    for i in range(div):
        if feat.seen[i]:
            free(feat.parts[i])
    free(feat.parts)
    free(feat.seen)
    free(feat)


cdef inline void update_square(size_t nr_class,
                               size_t now, double weight, size_t clas, SquareFeature* feat):
    cdef size_t part_idx = clas / LINE
    if not feat.seen[part_idx]:
        feat.parts[part_idx] = <DenseParams*>calloc(1, sizeof(DenseParams))
        feat.seen[part_idx] = True
    cdef DenseParams* params = feat.parts[part_idx]
    cdef size_t i = clas % LINE
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
        self.now = 0
        self.nr_raws = 20000
        self.raws = <DenseFeature**>malloc(self.nr_raws * sizeof(DenseFeature*))
        cdef size_t i
        for i in range(self.nr_raws):
            self.raws[i] = init_dense_feat(0, max_classes)
        self.n_corr = 0.0
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
                free_square_feat(<SquareFeature*>data.second, self.nr_class)
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
        for i in range(self.nr_raws):
            free(self.raws[i])
            self.raws[i] = init_dense_feat(0, self.nr_class)

    cdef int add_feature(self, uint64_t f) except -1:
        cdef size_t i
        cdef SquareFeature* feat = init_square_feat(f, self.nr_class)
        self.W[f] = <size_t>feat

    def batch_update(self, deltas):
        cdef size_t feat_addr
        self.now += 1
        for clas, feats in deltas.items():
            for f, d in feats.items():
                assert f != 0
                if d != 0:
                    feat_addr = self.W[f]
                    if feat_addr == 0:
                        self.add_feature(f)
                        feat_addr = self.W[f]
                    if feat_addr < self.nr_raws:
                        update_dense(self.now, d, clas, self.raws[feat_addr])
                    else:
                        update_square(self.nr_class,
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
                self.add_feature(f)
                feat_addr = self.W[f]
            if feat_addr < self.nr_raws:
                update_dense(self.now, 1.0, gold_i, self.raws[feat_addr])
                update_dense(self.now, -1.0, pred_i, self.raws[feat_addr])
            else:
                update_square(self.nr_class,
                              self.now, 1.0, gold_i, <SquareFeature*>feat_addr)
                update_square(self.nr_class,
                              self.now, -1.0, pred_i, <SquareFeature*>feat_addr)
   
    cdef inline int fill_scores(self, uint64_t* features, double* scores) except -1:
        cdef size_t feat_addr
        cdef uint64_t f

        cdef SquareFeature** active_square = self._active_square
        cdef DenseFeature** active_dense = self._active_dense
        cdef DenseFeature** raws = self.raws
        cdef size_t nr_square = 0
        cdef size_t nr_dense = 0
        cdef size_t nr_raws = self.nr_raws
        # First collect the active features, in two lots --- dense and square
        cdef size_t i = 0
        while features[i] != 0:
            f = features[i]
            i += 1
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
        cdef double inst_weight = 0
        for i in range(nr_class):
            scores[i] = 0
        for i in range(nr_dense):
            inst_weight += score_dense_feat(scores, active_dense[i])
        for i in range(nr_square):
            inst_weight += score_square_feat(scores, nr_class, active_square[i])
        #inst_weight = inst_weight / 2
        for i in range(nr_class):
            scores[i] += inst_weight

    def end_training(self, loc):
        cdef uint64_t f
        cdef double tmp
        cdef SquareFeature* feat
        cdef pair[uint64_t, size_t] data
        cdef dense_hash_map[uint64_t, size_t].iterator it
        it = self.W.begin()
        while it != self.W.end():
            data = deref(it)
            inc(it)
            if data.second >= self.nr_raws:
                feat = <SquareFeature*>data.second
                for i in range(get_div(self.nr_class)):
                    if feat.seen[i]:
                        params = feat.parts[i]
                        for j in range(LINE):
                            # Save the unaveraged value in accs
                            tmp = params.w[j]
                            params.acc[j] += (self.now - params.last_upd[j]) * params.w[j]
                            params.w[j] = params.acc[j] / self.now
                            params.acc[j] = tmp
        cdef DenseFeature* rfeat
        for i in range(1, self.nr_raws):
            rfeat = self.raws[i]
            weights = self.raws[i].w
            accs = self.raws[i].acc
            last_upd = self.raws[i].last_upd
            for c in range(self.nr_class):
                # Save the unaveraged value in accs so that we can easily unaverage
                tmp = weights[c]
                accs[c] += (self.now - last_upd[c]) * weights[c]
                weights[c] = accs[c] / self.now
                accs[c] = tmp
        self._save(loc)
    
    def _save(self, out_loc):
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
        cdef double unary_weight
        for nr_seen, feat_id in by_nr_seen:
            feat_addr = self.W[feat_id]
            non_zeroes = []
            unary_weight = 0.0
            if feat_addr == 0:
                continue
            elif feat_addr < self.nr_raws:
                for i in range(self.nr_class):
                    if self.raws[feat_addr].w[i] != 0:
                        unary_weight += self.raws[feat_addr].w[i]
                        non_zeroes.append('%d=%.3f' % (i, self.raws[feat_addr].w[i]))
            else:
                feat = <SquareFeature*>feat_addr
                for i in range(get_div(self.nr_class)):
                    if feat.seen[i]:
                        params = feat.parts[i]
                        for j in range(LINE):
                            clas = i * LINE + j
                            if params.w[j] != 0:
                                unary_weight += params.w[j]
                                non_zeroes.append('%d=%.3f ' % (clas, params.w[j]))
                 
            if non_zeroes:
                out.write(u'%d\t%d\t%.3f\t%s\n' % (feat_id, nr_seen, unary_weight,
                                                   ' '.join(non_zeroes)))
        out.close()

    def load(self, in_, size_t thresh=0):
        cdef SquareFeature* feat
        cdef size_t i, nr_seen
        nr_feat = 0
        nr_weight = 0
        #cdef uint64_t f, clas, idx
        in_ = gzip.open(in_, 'rb')
        header = in_.readline()
        self.nr_class = int(header.split()[1])
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
        cdef double unary_weight
        cdef size_t cls
        cdef char* token
        for py_line in in_:
            line = <char*>py_line
            token = strtok(line, '\t')
            f = strtoull(token, NULL, 10)
            token = strtok(NULL, '\t')
            nr_seen = atoi(token)
            token = strtok(NULL, '\t')
            unary_weight = atof(token)
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
                nr_weight += load_dense_feat(self.nr_class, unary_weight, weights,
                                             nr_seen, self.raws[nr_raws])
                self.raws[nr_raws].id = f
                self.W[f] = nr_raws
                nr_raws += 1
                nr_feat += 1
                continue
            else:
                self.add_feature(f)
                nr_weight += load_square_feat(self.nr_class, unary_weight, weights,
                                              nr_seen, <SquareFeature*>self.W[f])
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
            if f_id == 0:
                continue
            elif (feat_addr <= self.nr_raws):
                n_feats += 1
                continue
            feat = <SquareFeature*>feat_addr
            if feat.nr_seen < thresh:
                free_square_feat(feat, self.nr_class)
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
        cdef size_t max_raws = self.nr_raws
        cdef size_t length = len(q)
        if length < max_raws:
            return None
        cutoff = q[self.nr_raws][0]
        vacancies = []
        for i in range(1, self.nr_raws):
            if self.raws[i].id == 0:
                vacancies.append(i)
            elif self.raws[i].nr_seen < cutoff:
                assert self.raws[i].id != 0
                self._dense_to_square(self.raws[i].id, i)
                vacancies.append(i)
        for freq, f_id in q[:self.nr_raws]:
            if not vacancies:
                break
            assert f_id != 0
            f_addr = self.W[f_id]
            if f_addr > self.nr_raws:
                self._square_to_dense(f_id, f_addr, vacancies.pop())
        assert not vacancies, str(vacancies)

    def _dense_to_square(self, uint64_t f_id, size_t raw_idx):
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
        cdef size_t div = get_div(self.nr_class)
        for part_idx in range(div):
            for i in range(LINE):
                clas = (part_idx * LINE) + i
                if weights[clas] == 0:
                    continue
                if not feat.seen[part_idx]:
                    feat.parts[part_idx] = <DenseParams*>calloc(1, sizeof(DenseParams))
                    feat.seen[part_idx] = True
                params = feat.parts[part_idx]
                params.acc[i] = accs[clas]
                params.w[i] = weights[clas]
                params.last_upd[i] = last_upd[i]

    def _square_to_dense(self, uint64_t f_id, size_t feat_addr, size_t i):
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
        cdef size_t div = get_div(self.nr_class)
        for j in range(div):
            if feat.seen[j]:
                part_idx = j * LINE
                w = feat.parts[j].w
                accs = feat.parts[j].acc
                last_upd = feat.parts[j].last_upd
                for k in range(LINE):
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

        free_square_feat(feat, self.nr_class)
        self.W[f_id] = i
        assert self.W[f_id] < self.nr_raws
        assert raw.e <= self.nr_class, raw.e
