from libc.stdlib cimport strtoull, strtoul, atof
from libc.string cimport strtok

from cymem.cymem cimport Address

import random


DEF LINE_SIZE = 7


cdef WeightLine* new_weight_line(Pool mem, const C start) except NULL:
    cdef WeightLine* line = <WeightLine*>mem.alloc(1, sizeof(WeightLine))
    line.start = start
    return line


cdef CountLine* new_count_line(Pool mem, const C start) except NULL:
    cdef CountLine* line = <CountLine*>mem.alloc(1, sizeof(CountLine))
    line.start = start
    return line


cdef WeightLine** new_weight_matrix(Pool mem, C nr_class):
    cdef I nr_lines = get_nr_rows(nr_class)
    return <WeightLine**>mem.alloc(nr_lines, sizeof(WeightLine*))
 

cdef CountLine** new_count_matrix(Pool mem, C nr_class):
    cdef I nr_lines = get_nr_rows(nr_class)
    return <CountLine**>mem.alloc(nr_lines, sizeof(CountLine*))
 

cdef TrainFeat* new_train_feat(Pool mem, const C n) except NULL:
    cdef TrainFeat* output = <TrainFeat*>mem.alloc(1, sizeof(TrainFeat))
    output.weights = new_weight_matrix(mem, n)
    output.totals = new_weight_matrix(mem, n)
    output.counts = new_count_matrix(mem, n)
    output.times = new_count_matrix(mem, n)
    return output


cdef I get_row(const C clas):
    return clas / LINE_SIZE


cdef I get_col(const C clas):
    return clas % LINE_SIZE


cdef I get_nr_rows(const C n) except 0:
    cdef I nr_lines = get_row(n)
    if nr_lines == 0 or n % LINE_SIZE != 0:
        nr_lines += 1
    return nr_lines


cdef int update_weight(Pool mem, TrainFeat* feat, const C clas, const W inc) except -1:
    '''Update the weight for a parameter (a {feature, class} pair).'''
    cdef I row = get_row(clas)
    cdef I col = get_col(clas)
    if feat.weights[row] == NULL:
        feat.weights[row] = new_weight_line(mem, clas - col)
    feat.weights[row].line[col] += inc


cdef int update_accumulator(Pool mem, TrainFeat* feat, const C clas, const I time) except -1:
    '''Help a weight update for one (class, feature) pair for averaged models,
    e.g. Average Perceptron. Efficient averaging requires tracking the total
    weight for the feature, which requires a time-stamp so we can fast-forward
    through iterations where the weight was unchanged.'''
    cdef I row = get_row(clas)
    cdef I col = get_col(clas)
    if feat.weights[row] == NULL:
        feat.weights[row] = new_weight_line(mem, clas - col)
    if feat.totals[row] == NULL:
        feat.totals[row] = new_weight_line(mem, clas - col)
    if feat.times[row] == NULL:
        feat.times[row] = new_count_line(mem, clas - col)
    cdef W weight = feat.weights[row].line[col]
    cdef I unchanged = time - feat.times[row].line[col]
    feat.totals[row].line[col] += unchanged * weight
    feat.times[row].line[col] = time


cdef int update_count(Pool mem, TrainFeat* feat, const C clas, const I inc) except -1:
    '''Help a weight update for one (class, feature) pair by tracking how often
    the feature has been updated.  Used in Adagrad and others.
    '''
    cdef I row = get_row(clas)
    cdef I col = get_col(clas)
    if feat.counts[row] == NULL:
        feat.counts[row] = new_count_line(mem, clas - col)
    feat.counts[row].line[col] += inc


cdef int set_scores(W* scores, WeightLine* weight_lines, I nr_rows, C nr_class) except -1:
    cdef:
        I row
        I col
    cdef size_t start
    for row in range(nr_rows):
        start = weight_lines[row].start
        for col in range(0, min(nr_class - start, LINE_SIZE)):
            scores[weight_lines[row].start + col] += weight_lines[row].line[col]


cdef int average_weight(TrainFeat* feat, const C nr_class, const I time) except -1:
    cdef I unchanged
    cdef I row
    cdef I col
    for row in range(get_nr_rows(nr_class)):
        if feat.weights[row] == NULL:
            continue
        for col in range(LINE_SIZE):
            unchanged = (time + 1) - feat.times[row].line[col]
            feat.totals[row].line[col] += unchanged * feat.weights[row].line[col]
            feat.weights[row].line[col] = feat.totals[row].line[col] / time



cdef class LinearModel:
    def __cinit__(self, nr_class):
        self.total = 0
        self.n_corr = 0
        self.nr_class = nr_class
        self.time = 0
        self.weights = PointerMap()
        self.train_weights = PointerMap()
        self.mem = Pool()
        self.scores = <double*>self.mem.alloc(self.nr_class, sizeof(double))

    def __call__(self, list py_feats):
        feat_mem = Address(len(py_feats), sizeof(F))
        cdef F* features = <F*>feat_mem.addr
        cdef F feat
        for i, feat in enumerate(py_feats):
            features[i] = feat
        scores_mem = Address(self.nr_class, sizeof(double))
        scores = <double*>scores_mem.addr
        self.score(scores, features, len(py_feats))
        py_scores = []
        for i in range(self.nr_class):
            py_scores.append(scores[i])
        return py_scores

    cdef TrainFeat* new_feat(self, F feat_id) except NULL:
        cdef TrainFeat* feat = new_train_feat(self.mem, self.nr_class)
        self.weights.set(feat_id, feat.weights)
        self.train_weights.set(feat_id, feat)
        return feat

    cdef I gather_weights(self, WeightLine* w_lines, F* feat_ids, I nr_active) except *:
        cdef:
            WeightLine** feature
            I i, j
        
        cdef I nr_rows = get_nr_rows(self.nr_class)
        cdef I f_i = 0
        for i in range(nr_active):
            feature = <WeightLine**>self.weights.get(feat_ids[i])
            if feature != NULL:
                for row in range(nr_rows):
                    if feature[row] != NULL:
                        w_lines[f_i] = feature[row][0]
                        f_i += 1
        return f_i

    cdef int score(self, W* scores, F* features, I nr_active) except -1:
        cdef I nr_rows = nr_active * get_nr_rows(self.nr_class)
        cdef Address weights_mem = Address(nr_rows, sizeof(WeightLine))
        cdef WeightLine* weights = <WeightLine*>weights_mem.addr
        cdef I f_i = self.gather_weights(weights, features, nr_active)
        set_scores(scores, weights, f_i, self.nr_class)

    cpdef int update(self, dict updates) except -1:
        cdef C clas
        cdef F feat_id
        cdef TrainFeat* feat
        cdef double upd
        self.time += 1
        for clas, features in updates.items():
            for feat_id, upd in features.items():
                assert feat_id != 0
                feat = <TrainFeat*>self.train_weights.get(feat_id)
                if feat == NULL:
                    feat = self.new_feat(feat_id)
                update_accumulator(self.mem, feat, clas, self.time)
                update_count(self.mem, feat, clas, 1)
                update_weight(self.mem, feat, clas, upd)

    def end_training(self):
        cdef size_t i
        for i in range(self.train_weights.size):
            if self.train_weights.cells[i].key == 0:
                continue
            feat = <TrainFeat*>self.train_weights.cells[i].value
            average_weight(feat, self.nr_class, self.time)


    def end_train_iter(self, iter_num, feat_thresh):
        pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
        acc = pc(self.n_corr, self.total)
        msg = "#%d: Moves %d/%d=%s" % (iter_num, self.n_corr, self.total, acc)
        self.n_corr = 0
        self.total = 0

    def dump(self, file_):
        cdef F feat_id
        cdef C row
        cdef I i
        cdef C nr_rows = get_nr_rows(self.nr_class)
        for i in range(self.weights.size):
            if self.weights.cells[i].key == 0:
                continue
            feat_id = self.weights.cells[i].key
            feat = <WeightLine**>self.weights.cells[i].value
            for row in range(nr_rows):
                if feat[row] == NULL:
                    continue
                line = []
                line.append(feat_id)
                line.append(row)
                line.append(feat[row].start)
                for col in range(LINE_SIZE):
                    line.append(feat[row].line[col])
                file_.write('\t'.join([str(p) for p in line]))
                file_.write('\n')

    def load(self, file_):
        cdef F feat_id
        cdef C nr_rows, row, start
        cdef I col
        cdef bytes py_line
        cdef bytes token
        nr_rows = get_nr_rows(self.nr_class)
        for py_line in file_:
            line = <char*>py_line
            token = strtok(line, '\t')
            feat_id = strtoull(token, NULL, 10)
            token = strtok(NULL, '\t')
            row = strtoul(token, NULL, 10)
            token = strtok(NULL, '\t')
            start = strtoul(token, NULL, 10)
            if self.weights.get(feat_id) == NULL:
                self.weights.set(feat_id, self.mem.alloc(nr_rows, sizeof(WeightLine*)))
            feature = <WeightLine**>self.weights.get(feat_id)
            feature[row] = <WeightLine*>self.mem.alloc(1, sizeof(WeightLine))
            feature[row].start = start
            for col in range(LINE_SIZE):
                token = strtok(NULL, '\t')
                feature[row].line[col] = atof(token)
