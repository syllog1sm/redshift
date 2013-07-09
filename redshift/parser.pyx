"""
MALT-style dependency parser
"""
cimport cython
import random
import os.path
from os.path import join as pjoin

from libc.stdlib cimport malloc, free, calloc
from libc.string cimport memcpy, memset

from _state cimport *
import sh
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
              'allow_reduce': allow_reduce, 'reuse_idx': reuse_idx,
              'beam_width': beam_width, 'ngrams': ngrams,
              'add_clusters': add_clusters}
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
                  allow_reattach=False, allow_reduce=False,
                  reuse_idx=False, beam_width=1,
                  ngrams=None, add_clusters=False):
        self.model_dir = self.setup_model_dir(model_dir, clean)
        self.features = FeatureSet(mask_value=index.hashes.get_mask_value(),
                                   feat_set=feat_set, ngrams=ngrams,
                                   add_clusters=add_clusters)
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        self.beam_width = beam_width
        if clean == True:
            self.new_idx(self.model_dir, self.features.n)
        else:
            self.load_idx(self.model_dir, self.features.n)
        self.moves = TransitionSystem(allow_reattach=allow_reattach,
                                      allow_reduce=allow_reduce)
        self.guide = Perceptron(self.moves.max_class, pjoin(model_dir, 'model'))
        self.say_config()

    def setup_model_dir(self, loc, clean):
        # TODO: Replace this with normal shell
        if clean and os.path.exists(loc):
            sh.rm('-rf', loc)
        if os.path.exists(loc):
            assert os.path.isdir(loc)
        else:
            sh.mkdir(loc)
        #sh.git.log(n=1, _out=loc.join('version').open('wb'), _bg=True) 
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
        indices = range(sents.length)
        for n in range(n_iter):
            random.shuffle(indices)
            for i in indices:
                if DEBUG:
                    print ' '.join(sents.strings[i][0])
                self.train_one(n, sents.s[i])
            print_train_msg(n, self.guide.n_corr, self.guide.total, self.guide.cache.n_hit,
                            self.guide.cache.n_miss)
            self.guide.n_corr = 0
            self.guide.total = 0
            if n < 3:
                self.guide.reindex()
            if n % 2 == 1 and self.feat_thresh > 1:
                self.guide.prune(self.feat_thresh)
        self.guide.finalize()

    cdef int train_one(self, int iter_num, Sentence* sent) except -1:
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
        cdef size_t k = self.beam_width
        cdef Beam beam = Beam(self.moves, k, sent.length, upd_strat=self.train_alg)
        cdef size_t p_idx, i
        cdef double* scores
        cdef Kernel* kernel
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        while not beam.is_finished:
            self.guide.cache.flush()
            for p_idx in range(beam.bsize):
                kernel = beam.next_state(p_idx, sent.pos)
                beam_scores[p_idx] = self._predict(sent, kernel)
            beam.extend_states(beam_scores)
        sent.parse.n_moves = beam.t
        beam.fill_parse(sent.parse.moves, sent.pos, sent.parse.heads, sent.parse.labels,
                        sent.parse.sbd, sent.parse.edits)
        free(beam_scores)

    cdef int train_one(self, int iter_num, Sentence* sent) except -1:
        cdef size_t p_idx, i
        cdef Kernel* kernel
        cdef Kernel* gold_kernel
        cdef uint64_t* feats
        cdef int* costs
        cdef int cost
        cdef size_t* g_heads = sent.parse.heads
        cdef size_t* g_labels = sent.parse.labels
        cdef bint* g_edits = sent.parse.edits
        cdef size_t* g_pos = sent.pos
        cdef Violation violn
        cdef double* scores
        cdef bint cache_hit = False
        cdef size_t k = self.beam_width
        cdef Beam beam = Beam(self.moves, k, sent.length, upd_strat=self.train_alg)
        cdef Beam gold_beam = Beam(self.moves, k, sent.length, upd_strat=self.train_alg)
        beam_scores = <double**>malloc(beam.k * sizeof(double*))
        gold_scores = <double**>malloc(beam.k * sizeof(double*))
        cdef double delta = 0
        cdef double max_violn = 0
        cdef size_t t = 0
        cdef size_t* ghist = <size_t*>calloc(sent.length * 3, sizeof(size_t))
        cdef size_t* phist = <size_t*>calloc(sent.length * 3, sizeof(size_t))
        while not beam.is_finished:
            self.guide.cache.flush()
            for i in range(beam.bsize):
                kernel = beam.next_state(i, sent.pos)
                beam_scores[i] = self._predict(sent, kernel)
            beam.extend_states(beam_scores)
            for i in range(gold_beam.bsize):
                kernel = gold_beam.next_state(i, sent.pos)
                gold_scores[i] = self._predict(sent, kernel)
                costs = self.moves.get_costs(gold_beam.beam[i], g_pos, g_heads,
                                             g_labels, g_edits)
                for clas in range(self.moves.nr_class):
                    if costs[clas] != 0:
                        gold_beam.valid[i][clas] = -1
            gold_beam.extend_states(gold_scores)
            delta = beam.beam[0].score - gold_beam.beam[0].score
            if delta >= max_violn:
                max_violn = delta
                t = gold_beam.beam[0].t
                assert t < (sent.length * 3)
                memcpy(ghist, gold_beam.beam[0].history, t * sizeof(size_t))
                memcpy(phist, beam.beam[0].history, t * sizeof(size_t))
            #beam.check_violation()
            #if self.train_alg == 'early' and beam.violn != None:
            #    break
            if beam.beam[0].score == gold_beam.beam[0].score:
                self.guide.n_corr += 1
            self.guide.total += 1
        free(beam_scores)
        free(gold_scores)
        self.guide.total += beam.gold.t
        if t > 0:
            counted = self._count_feats(sent, t, phist, ghist)
            self.guide.batch_update(counted)
        free(ghist)
        free(phist)

    def say_config(self):
        beam_settings = (self.beam_width, self.train_alg)
        print 'Beam settings: k=%d; upd_strat=%s' % beam_settings

    cdef double* _predict(self, Sentence* sent, Kernel* kernel) except NULL:
        cdef bint cache_hit = False
        scores = self.guide.cache.lookup(sizeof(Kernel), kernel, &cache_hit)
        if not cache_hit:
            feats = self.features.extract(sent, kernel)
            self.guide.fill_scores(feats, scores)
        return scores

    cdef dict _count_feats(self, Sentence* sent, size_t t, size_t* phist, size_t* ghist):
        cdef size_t d, i, f
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(sent.length)
        cdef State* pred_state = init_state(sent.length)
        # Find where the states diverge
        for d in range(t):
            if ghist[d] == phist[d]:
                self.moves.transition(ghist[d], gold_state)
                self.moves.transition(phist[d], pred_state)
            else:
                break
        else:
            return {}
        cdef dict counts = {}
        for clas in range(self.moves.nr_class):
            counts[clas] = {}
        cdef double* scores = self.guide.scores
        for i in range(d, t):
            fill_kernel(gold_state, sent.pos)
            feats = self.features.extract(sent, &gold_state.kernel)
            clas_counts = counts[ghist[i]]
            f = 0
            while True:
                value = feats[f]
                f += 1
                if value == 0:
                    break
                if value not in clas_counts:
                    clas_counts[value] = 0
                clas_counts[value] += 1
            self.moves.transition(ghist[i], gold_state)
        free_state(gold_state)
        for i in range(d, t):
            fill_kernel(pred_state, sent.pos)
            feats = self.features.extract(sent, &pred_state.kernel)
            clas_counts = counts[phist[i]]
            f = 0
            while True:
                value = feats[f]
                f += 1 
                if value == 0:
                    break
                if value not in clas_counts:
                    clas_counts[value] = 0
                clas_counts[value] -= 1
            self.moves.transition(phist[i], pred_state)
        free_state(pred_state)
        return counts


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
 
    cdef int train_one(self, int iter_num, Sentence* sent) except -1:
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef size_t* g_labels = sent.parse.labels
        cdef size_t* g_heads = sent.parse.heads
        cdef bint* g_edits = sent.parse.edits
        cdef State* s = init_state(sent.length)
        cdef bint online = self.train_alg == 'online'
        cdef size_t pred
        cdef uint64_t* feats
        cdef size_t _ = 0
        while not s.is_finished:
            fill_kernel(s, sent.pos)
            self.moves.fill_valid(s, valid)
            feats = self.features.extract(sent, &s.kernel)
            pred = self._predict(feats, valid, &s.guess_labels[s.i])
            if online:
                costs = self.moves.get_costs(s, sent.pos, g_heads, g_labels, g_edits)
                gold = self._predict(feats, costs, &_) if costs[pred] != 0 else pred
            else:
                gold = self.moves.break_tie(s, sent.pos, g_heads, g_labels, g_edits)
            self.guide.update(pred, gold, feats, 1)
            if online and iter_num >= 2 and random.random() < FOLLOW_ERR_PC:
                self.moves.transition(pred, s)
            else:
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
