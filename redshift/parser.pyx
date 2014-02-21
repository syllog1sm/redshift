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
from sentence cimport Sentence
from sentence cimport PySentence
from sentence import get_labels
from sentence cimport PySentence, Sentence
from transitions cimport TransitionSystem, transition_to_str 
from beam cimport Beam
from tagger cimport BeamTagger

from features.extractor cimport Extractor
import _parse_features
from _parse_features cimport *

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
    use_sbd = params['use_sbd'] == 'True'
    l_labels = params['left_labels']
    r_labels = params['right_labels']
    beam_width = int(params['beam_width'])
    feat_set = params['feat_set']
    ngrams = []
    for ngram_str in params.get('ngrams', '-1').split(','):
        if ngram_str == '-1': continue
        ngrams.append(tuple([int(i) for i in ngram_str.split('_')]))
    auto_pos = params['auto_pos'] == 'True'
    params = {'clean': False, 'train_alg': train_alg,
              'feat_set': feat_set, 'feat_thresh': feat_thresh,
              'vocab_thresh': 1, 'allow_reattach': allow_reattach,
              'use_edit': use_edit, 'use_sbd': use_sbd, 
              'allow_reduce': allow_reduce,
              'reuse_idx': reuse_idx, 'beam_width': beam_width,
              'ngrams': ngrams,
              'auto_pos': auto_pos}
    if beam_width >= 2:
        parser = BeamParser(model_dir, **params)
    else:
        raise StandardError
    pos_tags = set([int(line.split()[-1]) for line in
                        open(pjoin(model_dir, 'pos'))])
    _, nr_label = parser.moves.set_labels(pos_tags, _parse_labels_str(l_labels),
                            _parse_labels_str(r_labels))
    
    parser.load()
    return parser


cdef class BaseParser:
    cdef Extractor extractor
    cdef Perceptron guide
    cdef TransitionSystem moves
    cdef BeamTagger tagger
    cdef object model_dir
    cdef bint auto_pos
    cdef size_t beam_width
    cdef object add_extra
    cdef object train_alg
    cdef int feat_thresh
    cdef object feat_set
    cdef object ngrams
    cdef uint64_t* _features
    cdef size_t* _context

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  feat_set="zhang",
                  feat_thresh=0, vocab_thresh=5,
                  allow_reattach=False, allow_reduce=False,
                  use_edit=False, use_sbd=False,
                  reuse_idx=False, beam_width=1,
                  ngrams=None, auto_pos=False):
        self.model_dir = self.setup_model_dir(model_dir, clean)
        self.feat_set = feat_set
        self.ngrams = ngrams if ngrams is not None else []
        templates = _parse_features.baseline_templates()
        match_feats = []
        #templates += _parse_features.ngram_feats(self.ngrams)
        if 'disfl' in self.feat_set:
            templates += _parse_features.disfl
            templates += _parse_features.new_disfl
            templates += _parse_features.suffix_disfl
            templates += _parse_features.extra_labels
            templates += _parse_features.clusters
            templates += _parse_features.edges
            match_feats = _parse_features.match_templates()
        if 'stack' in self.feat_set:
            templates += _parse_features.stack_second
        if 'hist' in self.feat_set:
            templates += _parse_features.history
        if 'bitags' in self.feat_set:
            templates += _parse_features.pos_bigrams()
        if 'pauses' in self.feat_set:
            templates += _parse_features.pauses
        self.extractor = Extractor(templates, match_feats)
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(_parse_features.context_size(), sizeof(size_t))
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        self.beam_width = beam_width
        if clean == True:
            self.new_idx(self.model_dir)
        else:
            self.load_idx(self.model_dir)
        self.moves = TransitionSystem(allow_reattach=allow_reattach,
                                      allow_reduce=allow_reduce,
                                      use_edit=use_edit,
                                      use_sbd=use_sbd)
        self.auto_pos = auto_pos
        self.say_config()
        self.guide = Perceptron(self.moves.max_class, pjoin(model_dir, 'model.gz'))
        self.tagger = BeamTagger(model_dir, clean=False, reuse_idx=True)

    def setup_model_dir(self, loc, clean):
        if clean and os.path.exists(loc):
            shutil.rmtree(loc)
        if os.path.exists(loc):
            assert os.path.isdir(loc)
        else:
            os.mkdir(loc)
        return loc

    def train(self, list sents, n_iter=15):
        cdef size_t i, j, n
        cdef list held_out_gold
        cdef list held_out_parse
        self.tagger.setup_classes(sents)
        move_classes, nr_label = self.moves.set_labels(*get_labels(sents))
        #self.features.set_nr_label(nr_label)
        self.guide.set_classes(range(move_classes))
        self.write_cfg(pjoin(self.model_dir, 'parser.cfg'))
        if self.beam_width >= 2:
            self.guide.use_cache = True
        indices = list(range(len(sents)))
        cdef PySentence py_sent
        if not DEBUG:
            # Extra trick: sort by sentence length for first iteration
            indices.sort(key=lambda i: sents[i].length)
        for n in range(n_iter):
            for i in indices:
                py_sent = sents[i]
                if self.auto_pos:
                    self.tagger.train_sent(py_sent.c_sent)
                if self.train_alg == 'static':
                    self.static_train(n, py_sent.c_sent)
                else:
                    self.dyn_train(n, py_sent.c_sent)
            if self.auto_pos:
                print_train_msg(n, self.tagger.guide.n_corr, self.tagger.guide.total,
                                0, 0)
            print_train_msg(n, self.guide.n_corr, self.guide.total, self.guide.cache.n_hit,
                            self.guide.cache.n_miss)
            self.guide.n_corr = 0
            self.guide.total = 0
            if n % 2 == 1 and self.feat_thresh > 1:
                self.guide.prune(self.feat_thresh)
                self.tagger.guide.prune(self.feat_thresh / 2)
            if n < 3:
                self.guide.reindex()
                self.tagger.guide.reindex()
            random.shuffle(indices)
        if self.auto_pos:
            self.tagger.guide.finalize()
        self.guide.finalize()

    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        raise NotImplementedError

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        raise NotImplementedError
    
    def add_parses(self, list sents):
        self.guide.nr_class = self.moves.nr_class
        cdef size_t i
        cdef PySentence sent
        for sent in sents:
            self.parse(sent.c_sent)

    cdef int parse(self, Sentence* sent) except -1:
        raise NotImplementedError

    def save(self):
        self.guide.save(pjoin(self.model_dir, 'model.gz'))
        self.tagger.save()

    def load(self):
        self.guide.load(pjoin(self.model_dir, 'model.gz'), thresh=self.feat_thresh)
        self.tagger.guide.load(pjoin(self.model_dir, 'tagger.gz'), thresh=self.feat_thresh)

    def new_idx(self, model_dir):
        index.hashes.init_word_idx(pjoin(model_dir, 'words'))
        index.hashes.init_pos_idx(pjoin(model_dir, 'pos'))
        index.hashes.init_label_idx(pjoin(model_dir, 'labels'))

    def load_idx(self, model_dir):
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
            cfg.write(u'use_sbd\t%s\n' % self.moves.use_edit)
            cfg.write(u'left_labels\t%s\n' % ','.join(self.moves.left_labels))
            cfg.write(u'right_labels\t%s\n' % ','.join(self.moves.right_labels))
            cfg.write(u'beam_width\t%d\n' % self.beam_width)
            cfg.write(u'auto_pos\t%s\n' % self.auto_pos)
            #if not self.features.ngrams:
            #    cfg.write(u'ngrams\t-1\n')
            #else:
            #    ngram_strs = ['_'.join([str(i) for i in ngram])
            #                  for ngram in self.features.ngrams]
            #    cfg.write(u'ngrams\t%s\n' % u','.join(ngram_strs))
            cfg.write(u'feat_set\t%s\n' % self.feat_set)

    def __dealloc__(self):
        pass

cdef class BeamParser(BaseParser):
    cdef int parse(self, Sentence* sent) except -1:
        cdef State* s
        cdef Beam beam = Beam(self.moves, self.beam_width, sent.length)
        cdef size_t p_idx
        cdef Kernel* kernel
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        if self.auto_pos:
            self.tagger.tag(sent)
        self.guide.cache.flush()
        while not beam.is_finished:
            for p_idx in range(beam.bsize):
                pred = <State*>beam.beam[p_idx]
                self.moves.fill_valid(pred, beam.valid[p_idx])
                fill_kernel(pred, sent.pos) 
                beam_scores[p_idx] = self._predict(sent, &pred.kernel)
            beam.extend_states(beam_scores)
        s = <State*>beam.beam[0]
        sent.parse.n_moves = s.t
        beam.fill_parse(sent.parse.moves, sent.pos, sent.parse.heads, sent.parse.labels,
                        sent.parse.sbd, sent.parse.edits)
        sent.parse.score = beam.beam[0].score
        free(beam_scores)

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t  i
        cdef Kernel* kernel
        cdef double* scores
        cdef Beam beam = Beam(self.moves, self.beam_width, sent.length)
        cdef State* gold = init_state(sent.length)
        cdef State* pred
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        cdef size_t* ghist = <size_t*>calloc(sent.length * 3, sizeof(size_t))
        cdef size_t* phist = <size_t*>calloc(sent.length * 3, sizeof(size_t))
        cdef double max_violn = 0
        cdef size_t t = 0
        # Backup pos tags
        cdef size_t* bu_tags
        if self.auto_pos:
            bu_tags = <size_t*>calloc(sent.length, sizeof(size_t))
            memcpy(bu_tags, sent.pos, sent.length * sizeof(size_t))
            self.tagger.tag(sent)
        self.guide.cache.flush()
        while not beam.is_finished:
            for i in range(beam.bsize):
                pred = <State*>beam.beam[i]
                self.moves.fill_valid(pred, beam.valid[i])
                fill_kernel(pred, sent.pos)
                beam_scores[i] = self._predict(sent, &pred.kernel)
            beam.extend_states(beam_scores)
            oracle = self.moves.break_tie(gold, sent.pos, sent.parse.heads,
                                          sent.parse.labels, sent.parse.edits,
                                          sent.parse.sbd)
            fill_kernel(gold, sent.pos)
            scores = self._predict(sent, &gold.kernel)
            gold.score += scores[oracle]
            self.moves.transition(oracle, gold)
            pred = <State*>beam.beam[0]
            if (pred.score - gold.score) >= max_violn:
                max_violn = pred.score - gold.score
                t = gold.t
                memcpy(ghist, gold.history, t * sizeof(size_t))
                memcpy(phist, pred.history, t * sizeof(size_t))
        if t == 0:
            self.guide.now += 1
            self.guide.n_corr += beam.t
            self.guide.total += beam.t
        else:
            counted = self._count_feats(sent, t, t, phist, ghist)
            self.guide.batch_update(counted)
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags)
        free(ghist)
        free(phist)
        free(beam_scores)
        free_state(gold)

    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t i
        cdef Kernel* kernel
        cdef int* costs
        cdef State* p
        cdef State* g
        cdef size_t* bu_tags
        if self.auto_pos:
            bu_tags = <size_t*>calloc(sent.length, sizeof(size_t))
            memcpy(bu_tags, sent.pos, sent.length * sizeof(size_t))
            self.tagger.tag(sent)
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
                p = <State*>pred.beam[i]
                self.moves.fill_valid(p, pred.valid[i])
                fill_kernel(p, sent.pos)
                pred_scores[i] = self._predict(sent, &p.kernel)
                costs = self.moves.get_costs(p, sent.pos, sent.parse.heads,
                                             sent.parse.labels, sent.parse.edits,
                                             sent.parse.sbd)
                memcpy(pred.costs[i], costs, sizeof(int) * self.moves.nr_class)
            pred.extend_states(pred_scores)
            for i in range(gold.bsize):
                g = <State*>gold.beam[i]
                self.moves.fill_valid(g, gold.valid[i])
                fill_kernel(g, sent.pos)
                gold_scores[i] = self._predict(sent, &g.kernel)
                costs = self.moves.get_costs(<State*>gold.beam[i], sent.pos,
                                             sent.parse.heads, sent.parse.labels,
                                             sent.parse.edits, sent.parse.sbd)
                for clas in range(self.moves.nr_class):
                    if costs[clas] != 0:
                        gold.valid[i][clas] = -1
            gold.extend_states(gold_scores)
            g = <State*>gold.beam[0]
            p = <State*>pred.beam[0]
            delta = p.score - g.score
            if delta >= max_violn and p.cost >= 1:
                max_violn = delta
                pt = p.t
                gt = g.t
                memcpy(phist, p.history, p.t * sizeof(size_t))
                memcpy(ghist, g.history, g.t * sizeof(size_t))
            self.guide.n_corr += p.history[p.t-1] == g.history[g.t-1]
            self.guide.total += 1
        if max_violn >= 0:
            counted = self._count_feats(sent, pt, gt, phist, ghist)
            self.guide.batch_update(counted)
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags)
        free(ghist)
        free(phist)
        free(pred_scores)
        free(gold_scores)

    def say_config(self):
        beam_settings = (self.beam_width, self.train_alg)
        print 'Beam settings: k=%d; upd_strat=%s' % beam_settings
        if self.moves.allow_reattach and self.moves.allow_reduce:
            print 'NM L+D'
        elif self.moves.allow_reattach:
            print 'NM L'
        elif self.moves.allow_reduce:
            print 'NM D'
        else:
            print 'NM None'

    cdef double* _predict(self, Sentence* sent, Kernel* kernel) except NULL:
        cdef bint cache_hit = False
        scores = self.guide.cache.lookup(sizeof(Kernel), kernel, &cache_hit)
        if not cache_hit:
            fill_context(self._context, self.moves.n_labels, sent, kernel)
            self.extractor.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
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
            if i < gt:
                fill_kernel(gold_state, sent.pos)
                self._inc_feats(counts[ghist[i]], sent, &gold_state.kernel, g_inc)
                self.moves.transition(ghist[i], gold_state)
            if i < pt:
                fill_kernel(pred_state, sent.pos)
                self._inc_feats(counts[phist[i]], sent, &pred_state.kernel, p_inc)
                self.moves.transition(phist[i], pred_state)
        free_state(gold_state)
        free_state(pred_state)
        return counts

    cdef int _inc_feats(self, dict counts, Sentence* sent, Kernel* k, double inc) except -1:
        fill_context(self._context, self.moves.n_labels, sent, k)
        self.extractor.extract(self._features, self._context)
 
        cdef size_t f = 0
        while self._features[f] != 0:
            if self._features[f] not in counts:
                counts[self._features[f]] = 0
            counts[self._features[f]] += inc
            f += 1


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
