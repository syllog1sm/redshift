"""
MALT-style dependency parser
"""
cimport cython
import random
import os.path
from os.path import join as pjoin
import shutil

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

from _state cimport *
from io_parse cimport Sentence, Sentences
from transitions cimport TransitionSystem, transition_to_str 
from beam cimport Beam, Violation
from features cimport FeatureSet

import index.hashes
cimport index.hashes

from learn.perceptron cimport Perceptron

from libc.stdint cimport uint64_t, int64_t


VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50


DEBUG = False 
def set_debug(val):
    global DEBUG
    DEBUG = val


def load_parser(model_dir, reuse_idx=False):
    params = dict([line.split() for line in open(pjoin(model_dir, 'parser.cfg'))])
    train_alg = params['train_alg']
    feat_thresh = int(params['feat_thresh'])
    allow_reattach = params['allow_reattach'] == 'True'
    allow_reduce = params['allow_reduce'] == 'True'
    use_edit = params['use_edit'] == 'True'
    l_labels = params['left_labels']
    r_labels = params['right_labels']
    beam_width = int(params['beam_width'])
    feat_set = params['feat_set']
    ngrams = []
    for ngram_str in params['ngrams'].split(','):
        if ngram_str == '-1': continue
        ngrams.append(tuple([int(i) for i in ngram_str.split('_')]))
    add_clusters = params['add_clusters'] == 'True'
    params = {'clean': False, 'train_alg': train_alg,
              'feat_set': feat_set, 'feat_thresh': feat_thresh,
              'vocab_thresh': 1, 'allow_reattach': allow_reattach,
              'use_edit': use_edit, 'allow_reduce': allow_reduce,
              'reuse_idx': reuse_idx, 'beam_width': beam_width,
              'ngrams': ngrams, 'add_clusters': add_clusters}
    if beam_width >= 2:
        parser = BeamParser(model_dir, **params)
    else:
        parser = GreedyParser(model_dir, **params)
    pos_tags = set([int(line.split()[-1]) for line in
                        open(pjoin(model_dir, 'pos'))])
    # TODO: Fix this
    _, nr_label = parser.moves.set_labels(pos_tags, _parse_labels_str(l_labels),
                            _parse_labels_str(r_labels))
    parser.features.set_nr_label(nr_label)

    parser.load()
    return parser


cdef class BaseParser:
    cdef FeatureSet features
    cdef Perceptron guide
    cdef object model_dir
    cdef size_t beam_width
    cdef TransitionSystem moves
    cdef object add_extra
    cdef object train_alg
    cdef int feat_thresh

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  feat_set="zhang",
                  feat_thresh=0, vocab_thresh=5,
                  allow_reattach=False, allow_reduce=False, use_edit=False,
                  reuse_idx=False, beam_width=1,
                  ngrams=None, add_clusters=False):
        self.model_dir = self.setup_model_dir(model_dir, clean)
        self.features = FeatureSet(feat_set=feat_set, ngrams=ngrams,
                                   add_clusters=add_clusters)
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        self.beam_width = beam_width
        if clean == True:
            self.new_idx(self.model_dir, self.features.n)
        else:
            self.load_idx(self.model_dir, self.features.n)
        self.moves = TransitionSystem(allow_reattach=allow_reattach,
                                      allow_reduce=allow_reduce, use_edit=use_edit)
        self.guide = Perceptron(self.moves.max_class, pjoin(model_dir, 'model'))
        self.say_config()

    def setup_model_dir(self, loc, clean):
        if clean and os.path.exists(loc):
            shutil.rmtreeloc)
        if os.path.exists(loc):
            assert os.path.isdir(loc)
        else:
            os.mkdir(loc)
        return loc

    def train(self, Sentences sents, n_iter=15):
        cdef size_t i, j, n
        cdef Sentence* sent
        cdef Sentences held_out_gold
        cdef Sentences held_out_parse
        move_classes, nr_label = self.moves.set_labels(*sents.get_labels())
        self.features.set_nr_label(nr_label)
        self.guide.set_classes(range(move_classes))
        self.write_cfg(pjoin(self.model_dir, 'parser.cfg'))
        if self.beam_width >= 2:
            self.guide.use_cache = True
        indices = list(range(sents.length))
        if not DEBUG:
            # Extra trick: sort by sentence length for first iteration
            indices.sort(key=lambda i: sents.s[i].length)
        for n in range(n_iter):
            for i in indices:
                if DEBUG:
                    print ' '.join(sents.strings[i][0])
                if self.train_alg == 'static':
                    self.static_train(n, sents.s[i])
                else:
                    self.dyn_train(n, sents.s[i])
            print_train_msg(n, self.guide.n_corr, self.guide.total, self.guide.cache.n_hit,
                            self.guide.cache.n_miss)
            self.guide.n_corr = 0
            self.guide.total = 0
            if n % 2 == 1 and self.feat_thresh > 1:
                self.guide.prune(self.feat_thresh)
            if n < 3:
                self.guide.reindex()
            random.shuffle(indices)
        self.guide.finalize()

    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        raise NotImplementedError

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        raise NotImplementedError
    
    def add_parses(self, Sentences sents):
        self.guide.nr_class = self.moves.nr_class
        cdef size_t i
        for i in range(sents.length):
            if DEBUG:
                print ' '.join(sents.strings[i][0])
            self.parse(sents.s[i])

    cdef int parse(self, Sentence* sent) except -1:
        raise NotImplementedError

    def save(self):
        self.guide.save(pjoin(self.model_dir, 'model'))

    def load(self):
        self.guide.load(pjoin(self.model_dir, 'model'), thresh=self.feat_thresh)

    def new_idx(self, model_dir, size_t n_predicates):
        index.hashes.init_word_idx(pjoin(model_dir, 'words'))
        index.hashes.init_pos_idx(pjoin(model_dir, 'pos'))
        index.hashes.init_label_idx(pjoin(model_dir, 'labels'))

    def load_idx(self, model_dir, size_t n_predicates):
        index.hashes.load_word_idx(pjoin(model_dir, 'words'))
        index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        index.hashes.load_label_idx(pjoin(model_dir, 'labels'))
   
    def write_cfg(self, loc):
        with open(loc, 'w') as cfg:
            cfg.write(u'model_dir\t%s\n' % self.model_dir)
            cfg.write(u'train_alg\t%s\n' % self.train_alg)
            cfg.write(u'feat_thresh\t%d\n' % self.feat_thresh)
            cfg.write(u'allow_reattach\t%s\n' % self.moves.allow_reattach)
            cfg.write(u'allow_reduce\t%s\n' % self.moves.allow_reduce)
            cfg.write(u'use_edit\t%s\n' % self.moves.use_edit)
            cfg.write(u'left_labels\t%s\n' % ','.join(self.moves.left_labels))
            cfg.write(u'right_labels\t%s\n' % ','.join(self.moves.right_labels))
            cfg.write(u'beam_width\t%d\n' % self.beam_width)
            if not self.features.ngrams:
                cfg.write(u'ngrams\t-1\n')
            else:
                ngram_strs = ['_'.join([str(i) for i in ngram])
                              for ngram in self.features.ngrams]
                cfg.write(u'ngrams\t%s\n' % u','.join(ngram_strs))
            cfg.write(u'feat_set\t%s\n' % self.features.name)
            cfg.write(u'add_clusters\t%s\n' % self.features.add_clusters)

    def __dealloc__(self):
        pass

cdef class BeamParser(BaseParser):
    cdef int parse(self, Sentence* sent) except -1:
        cdef Beam beam = Beam(self.moves, self.beam_width, sent.length)
        cdef size_t p_idx
        cdef Kernel* kernel
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        self.guide.cache.flush()
        while not beam.is_finished:
            for p_idx in range(beam.bsize):
                kernel = beam.next_state(p_idx, sent.pos)
                beam_scores[p_idx] = self._predict(sent, kernel)
            beam.extend_states(beam_scores)
        sent.parse.n_moves = beam.t
        beam.fill_parse(sent.parse.moves, sent.pos, sent.parse.heads, sent.parse.labels,
                        sent.parse.sbd, sent.parse.edits)
        free(beam_scores)

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t  i
        cdef Kernel* kernel
        cdef double* scores
        cdef Beam beam = Beam(self.moves, self.beam_width, sent.length)
        cdef State* gold = init_state(sent.length)
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        cdef size_t* ghist = <size_t*>calloc(sent.length * 3, sizeof(size_t))
        cdef size_t* phist = <size_t*>calloc(sent.length * 3, sizeof(size_t))
        cdef double max_violn = 0
        cdef size_t t = 0
        while not beam.is_finished:
            self.guide.cache.flush()
            for i in range(beam.bsize):
                kernel = beam.next_state(i, sent.pos)
                beam_scores[i] = self._predict(sent, kernel)
            beam.extend_states(beam_scores)
            oracle = self.moves.break_tie(gold, sent.pos, sent.parse.heads,
                                          sent.parse.labels, sent.parse.edits)
            fill_kernel(gold, sent.pos)
            scores = self._predict(sent, &gold.kernel)
            gold.score += scores[oracle]
            self.moves.transition(oracle, gold)
            if (beam.beam[0].score - gold.score) >= max_violn:
                max_violn = beam.beam[0].score - gold.score
                t = gold.t
                memcpy(ghist, gold.history, t * sizeof(size_t))
                memcpy(phist, beam.beam[0].history, t * sizeof(size_t))
        if t == 0:
            self.guide.n_corr += beam.t
            self.guide.total += beam.t
        else:
            counted = self._count_feats(sent, t, t, phist, ghist)
            self.guide.batch_update(counted)
        free(ghist)
        free(phist)
        free(beam_scores)
        free_state(gold)

    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t i
        cdef Kernel* kernel
        cdef int* costs
        
        ghist = <size_t*>malloc(sent.length * 3 * sizeof(size_t))
        phist = <size_t*>malloc(sent.length * 3 * sizeof(size_t))
        for i in range(sent.length * 3):
            ghist[i] = self.moves.nr_class
            phist[i] = self.moves.nr_class
        pred = Beam(self.moves, self.beam_width, sent.length)
        gold = Beam(self.moves, self.beam_width, sent.length)
        pred_scores = <double**>malloc(self.beam_width * sizeof(double*))
        gold_scores = <double**>malloc(self.beam_width * sizeof(double*))
        cdef double delta = 0
        cdef double max_violn = -1
        cdef size_t pt = 0
        cdef size_t gt = 0
        self.guide.cache.flush()
        while not pred.is_finished and not gold.is_finished:
            for i in range(pred.bsize):
                kernel = pred.next_state(i, sent.pos)
                pred_scores[i] = self._predict(sent, kernel)
                costs = self.moves.get_costs(pred.beam[i], sent.pos, sent.parse.heads,
                                             sent.parse.labels, sent.parse.edits)
                memcpy(pred.costs[i], costs, sizeof(int) * self.moves.nr_class)
            pred.extend_states(pred_scores)
            for i in range(gold.bsize):
                kernel = gold.next_state(i, sent.pos)
                gold_scores[i] = self._predict(sent, kernel)
                costs = self.moves.get_costs(gold.beam[i], sent.pos,
                                             sent.parse.heads, sent.parse.labels,
                                             sent.parse.edits)
                for clas in range(self.moves.nr_class):
                    if costs[clas] != 0:
                        gold.valid[i][clas] = -1
            gold.extend_states(gold_scores)
            delta = pred.beam[0].score - gold.beam[0].score
            if delta >= max_violn and pred.beam[0].cost >= 1:
                max_violn = delta
                pt = pred.beam[0].t
                gt = gold.beam[0].t
                memcpy(phist, pred.beam[0].history, pt * sizeof(size_t))
                memcpy(ghist, gold.beam[0].history, gt * sizeof(size_t))
        if max_violn < 0:
            self.guide.n_corr += pred.beam[0].t
            self.guide.total += pred.beam[0].t
        else:
            counted = self._count_feats(sent, pt, gt, phist, ghist)
            self.guide.batch_update(counted)
        free(ghist)
        free(phist)
        free(pred_scores)
        free(gold_scores)

    def say_config(self):
        beam_settings = (self.beam_width, self.train_alg)
        print 'Beam settings: k=%d; upd_strat=%s' % beam_settings
        print 'Edits=%s' % self.moves.use_edit

    cdef double* _predict(self, Sentence* sent, Kernel* kernel) except NULL:
        cdef bint cache_hit = False
        scores = self.guide.cache.lookup(sizeof(Kernel), kernel, &cache_hit)
        if not cache_hit:
            feats = self.features.extract(sent, kernel)
            self.guide.fill_scores(feats, scores)
        return scores

    cdef dict _count_feats(self, Sentence* sent, size_t pt, size_t gt,
                           size_t* phist, size_t* ghist):
        cdef size_t d, i, f
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(sent.length)
        cdef State* pred_state = init_state(sent.length)
        # Find where the states diverge
        cdef dict counts = {}
        for clas in range(self.moves.nr_class):
            counts[clas] = {}
        cdef bint seen_diff = False
        #g_inc = float(pt) / float(gt)
        #p_inc = - (float(gt) / float(pt))
        g_inc = 1.0
        p_inc = -1.0
        for i in range(max((pt, gt))):
            self.guide.total += 1
            if not seen_diff and ghist[i] == phist[i]:
                self.guide.n_corr += 1
                self.moves.transition(ghist[i], gold_state)
                self.moves.transition(phist[i], pred_state)
                continue
            seen_diff = True
            if not gold_state.is_finished:
                fill_kernel(gold_state, sent.pos)
                self._inc_feats(counts[ghist[i]], sent, &gold_state.kernel, g_inc)
                self.moves.transition(ghist[i], gold_state)
            if not pred_state.is_finished:
                fill_kernel(pred_state, sent.pos)
                self._inc_feats(counts[phist[i]], sent, &pred_state.kernel, p_inc)
                self.moves.transition(phist[i], pred_state)
        free_state(gold_state)
        free_state(pred_state)
        return counts

    cdef int _inc_feats(self, dict counts, Sentence* sent, Kernel* k, double inc) except -1:
        cdef uint64_t* feats = self.features.extract(sent, k)
        cdef size_t f = 0
        while feats[f] != 0:
            if feats[f] not in counts:
                counts[feats[f]] = 0
            counts[feats[f]] += inc
            f += 1

cdef double FOLLOW_ERR_PC = 0.90

cdef class GreedyParser(BaseParser):
    cdef int parse(self, Sentence* sent) except -1:
        cdef State* s
        cdef uint64_t* feats
        s = init_state(sent.length)
        sent.parse.n_moves = 0
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            feats = self.features.extract(sent, &s.kernel)
            self.moves.fill_valid(s, self.moves._costs)
            clas = self._predict(feats, self.moves._costs,
                                &s.guess_labels[s.i])
            self.moves.transition(clas, s)
        # No need to copy heads for root and start symbols
        cdef size_t i
        for i in range(1, sent.length - 1):
            sent.parse.heads[i] = s.heads[i]
            sent.parse.labels[i] = s.labels[i]
        fill_edits(s, sent.parse.edits)
        free_state(s)
 
    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef State* s = init_state(sent.length)
        cdef size_t pred
        cdef uint64_t* feats
        cdef size_t _ = 0
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            self.moves.fill_valid(s, valid)
            feats = self.features.extract(sent, &s.kernel)
            pred = self._predict(feats, valid, &s.guess_labels[s.i])
            costs = self.moves.get_costs(s, sent.pos, sent.parse.heads,
                                         sent.parse.labels, sent.parse.edits)
            gold = pred if costs[pred] == 0 else self._predict(feats, costs, &_)
            self.guide.update(pred, gold, feats, 1)
            if iter_num >= 2 and random.random() < FOLLOW_ERR_PC:
                self.moves.transition(pred, s)
            else:
                self.moves.transition(gold, s)
            self.guide.n_corr += (gold == pred)
            self.guide.total += 1
        free_state(s)
        free(valid)

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef State* s = init_state(sent.length)
        cdef size_t pred
        cdef uint64_t* feats
        cdef size_t _ = 0
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            feats = self.features.extract(sent, &s.kernel)
            self.moves.fill_valid(s, valid)
            pred = self._predict(feats, valid, &s.guess_labels[s.i])
            gold = self.moves.break_tie(s, sent.pos, sent.parse.heads,
                                         sent.parse.labels, sent.parse.edits)
            self.guide.update(pred, gold, feats, 1)
            self.moves.transition(gold, s)
            self.guide.n_corr += (gold == pred)
            self.guide.total += 1
        free_state(s)
        free(valid)

    def say_config(self):
        if self.moves.allow_reattach and self.moves.allow_reduce:
            print 'NM L+D'
        elif self.moves.allow_reattach:
            print 'NM L'
        elif self.moves.allow_reduce:
            print 'NM D'
   
    cdef int _predict(self, uint64_t* feats, int* valid, size_t* rlabel) except -1:
        cdef:
            size_t i
            double score
            size_t clas, best_valid, best_right
            double* scores

        cdef size_t right_move = 0
        cdef double valid_score = -10000
        cdef double right_score = -10000
        scores = self.guide.scores
        self.guide.fill_scores(feats, scores)
        seen_valid = False
        for clas in range(self.guide.nr_class):
            score = scores[clas]
            if valid[clas] == 0:
                if score > valid_score:
                    best_valid = clas
                    valid_score = score
                if not seen_valid:
                    seen_valid = True
            if self.moves.r_end > clas >= self.moves.r_start and score > right_score:
                best_right = clas
                right_score = score
        assert seen_valid 
        rlabel[0] = self.moves.labels[best_right]
        return best_valid


def print_train_msg(n, n_corr, n_move, n_hit, n_miss):
    pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
    move_acc = pc(n_corr, n_move)
    cache_use = pc(n_hit, n_hit + n_miss + 1e-100)
    msg = "#%d: Moves %d/%d=%s" % (n, n_corr, n_move, move_acc)
    if cache_use != 0:
        msg += '. Cache use %s' % cache_use
    print msg


def _parse_labels_str(labels_str):
    return [index.hashes.encode_label(l) for l in labels_str.split(',')]
