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
from sentence import get_labels
from sentence cimport Input, Sentence, AnswerToken
from transitions cimport Transition, transition, fill_valid, fill_costs
from transitions cimport get_nr_moves, fill_moves
from beam cimport Beam
#from tagger cimport BeamTagger

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
              'vocab_thresh': 1, 
              'beam_width': beam_width,
              'ngrams': ngrams,
              'auto_pos': auto_pos}
    if beam_width >= 2:
        parser = Parser(model_dir, **params)
    else:
        raise StandardError
    pos_tags = set([int(line.split()[0]) for line in
                        open(pjoin(model_dir, 'pos'))])
    #_, nr_label = parser.moves.set_labels(pos_tags, _parse_labels_str(l_labels),
    #                        _parse_labels_str(r_labels))
    
    parser.load()
    return parser


cdef class Parser:
    cdef Extractor extractor
    cdef Perceptron guide
    #cdef BeamTagger tagger
    cdef object model_dir
    cdef bint auto_pos
    cdef size_t beam_width
    cdef object add_extra
    cdef object train_alg
    cdef int feat_thresh
    cdef object feat_set
    cdef object ngrams
    cdef Transition* moves
    cdef uint64_t* _features
    cdef size_t* _context

    def __cinit__(self, model_dir, clean=False, train_alg='static',
                  feat_set="zhang",
                  feat_thresh=0, vocab_thresh=5,
                  beam_width=1,
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
        elif 'clusters' in self.feat_set:
            templates += _parse_features.clusters
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
        self.auto_pos = auto_pos
        self.guide = Perceptron(500, pjoin(model_dir, 'model.gz'))
        #self.tagger = BeamTagger(model_dir, clean=False, reuse_idx=True)

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
        #self.tagger.setup_classes(sents)
        #self.features.set_nr_label(nr_label)
        tags, left_labels, right_labels = get_labels(sents)
        self.nr_moves = get_nr_moves(left_labels, right_labels)
        self.moves = <Transition*>calloc(self.nr_moves, sizeof(Transition))
        self.guide.set_classes(range(self.nr_class))
        self.write_cfg(pjoin(self.model_dir, 'parser.cfg'))
        if self.beam_width >= 2:
            self.guide.use_cache = True
        indices = list(range(len(sents)))
        cdef Input py_sent
        if not DEBUG:
            # Extra trick: sort by sentence length for first iteration
            indices.sort(key=lambda i: sents[i].length)
        for n in range(n_iter):
            for i in indices:
                py_sent = sents[i]
                #if self.auto_pos:
                #    self.tagger.train_sent(py_sent.c_sent)
                self.train_sent(py_sent.c_sent, py_sent.c_sent.answer)
            self.guide.end_train_iter()
            random.shuffle(indices)
        #if self.auto_pos:
        #    self.tagger.guide.finalize()
        self.guide.finalize()

    cdef int train_sent(self, Sentence* sent, AnswerToken* gold_parse) except -1:
        cdef size_t i
        cdef size_t nr_move = sent.n * 3
        cdef Transition[500] g_hist
        cdef Transition[500] p_hist
        for i in range(sent.n * 3):
            g_hist[i] = self.nr_class
            p_hist[i] = self.nr_class
        p_beam = Beam(self.beam_width, sent.n, self.nr_class)
        g_beam = Beam(self.beam_width, sent.n, self.nr_class)
        cdef double delta = 0
        cdef double max_violn = -1
        cdef size_t pt = 0
        cdef size_t gt = 0
        cdef State* p
        cdef State* g
        self.guide.cache.flush()
        while not p_beam.is_finished and not g_beam.is_finished:
            for i in range(p_beam.bsize):
                self._predict(p_beam.beam[i], p_beam.moves[i], sent.steps)
                # Fill costs so we can see whether the prediction is gold-standard
                fill_costs(p_beam.beam[i], p_beam.moves[i], self.nr_class, gold_parse)
                # The False flag tells it to allow non-gold predictions
                p_beam.enqueue(i, False)
            p_beam.extend()
            for i in range(g_beam.bsize):
                self._predict(g_beam.beam[i], g_beam.moves[i], sent.steps)
                # Constrain this beam to only gold candidates
                fill_costs(g_beam.beam[i], g_beam.moves[i], self.nr_class, gold_parse)
                g_beam.enqueue(i, True)
            g_beam.extend()
            g = g_beam.beam[0]; p = p_beam.beam[0] 
            delta = p.score - g.score
            if delta >= max_violn and p.cost >= 1:
                max_violn = delta
                pt = p.m
                gt = g.m
                memcpy(p_hist, p.history, pt * sizeof(size_t))
                memcpy(g_hist, g.history, gt * sizeof(size_t))
            self.guide.n_corr += p.history[p.m-1].clas == g.history[g.m-1].clas
            self.guide.total += 1
        if max_violn >= 0:
            counted = self._count_feats(sent, pt, gt, p_hist, g_hist)
            self.guide.batch_update(counted)
            # TODO: We should tick the epoch here if max_violn == 0, right?

    cdef int _predict(self, State* s, Transition* classes, Step* steps):
        cdef bint cache_hit = False
        fill_slots(s)
        scores = self.guide.cache.lookup(sizeof(SlotTokens), &s.slots, &cache_hit)
        if not cache_hit:
            fill_context(self._context, &s.slots, s.parse, steps)
            self.extractor.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
        fill_valid(s, classes, self.nr_class)
        cdef size_t i
        for i in range(self.nr_class):
            classes[i].score = scores[i]

    cdef dict _count_feats(self, Sentence* sent, size_t pt, size_t gt,
                           Transition* phist, Transition* ghist):
        cdef size_t d, i, f
        cdef uint64_t* feats
        cdef size_t clas
        cdef State* gold_state = init_state(sent.n)
        cdef State* pred_state = init_state(sent.n)
        # Find where the states diverge
        cdef dict counts = {}
        for clas in range(self.nr_class):
            counts[clas] = {}
        cdef bint seen_diff = False
        g_inc = 1.0
        p_inc = -1.0
        for i in range(max((pt, gt))):
            self.guide.total += 1
            if not seen_diff and ghist[i].clas == phist[i].clas:
                self.guide.n_corr += 1
                transition(&ghist[i], gold_state)
                transition(&phist[i], pred_state)
                continue
            seen_diff = True
            if i < gt:
                self._inc_feats(counts[ghist[i]], gold_state, sent.steps, g_inc)
                transition(&ghist[i], gold_state)
            if i < pt:
                self._inc_feats(counts[phist[i]], pred_state, sent.steps, p_inc)
                transition(&phist[i], pred_state)
        free_state(gold_state)
        free_state(pred_state)
        return counts

    cdef int _inc_feats(self, dict counts, State* s, Step* steps, double inc) except -1:
        fill_slots(s)
        fill_context(self._context, &s.slots, s.parse, steps)
        self.extractor.extract(self._features, self._context)
 
        cdef size_t f = 0
        while self._features[f] != 0:
            if self._features[f] not in counts:
                counts[self._features[f]] = 0
            counts[self._features[f]] += inc
            f += 1

    def add_parses(self, list sents):
        self.guide.nr_class = self.nr_class
        cdef size_t i
        cdef Input sent
        for sent in sents:
            self.parse(sent.c_sent)

    cdef int parse(self, Sentence* sent) except -1:
        cdef Beam beam = Beam(self.nr_class, self.beam_width, sent.n)
        cdef size_t p_idx
        #if self.auto_pos:
        #    self.tagger.tag(sent)
        self.guide.cache.flush()
        while not beam.is_finished:
            for i in range(beam.bsize):
                self._predict(beam.beam[i], beam.moves[i], sent.steps)
                # The False flag tells it to allow non-gold predictions
                beam.enqueue(i, False)
            beam.extend()
        beam.fill_parse(sent.answer)
        sent.score = beam.beam[0].score

    def save(self):
        self.guide.save(pjoin(self.model_dir, 'model.gz'))
        index.hashes.save_pos_idx(pjoin(self.model_dir, 'pos'))
        index.hashes.save_label_idx(pjoin(self.model_dir, 'labels'))
        #self.tagger.save()

    def load(self):
        self.guide.load(pjoin(self.model_dir, 'model.gz'), thresh=self.feat_thresh)
        #self.tagger.guide.load(pjoin(self.model_dir, 'tagger.gz'), thresh=self.feat_thresh)
        index.hashes.load_pos_idx(pjoin(self.model_dir, 'pos'))
        index.hashes.load_label_idx(pjoin(self.model_dir, 'labels'))
   
    def write_cfg(self, loc):
        with open(loc, 'w') as cfg:
            cfg.write(u'model_dir\t%s\n' % self.model_dir)
            cfg.write(u'train_alg\t%s\n' % self.train_alg)
            cfg.write(u'feat_thresh\t%d\n' % self.feat_thresh)
            #cfg.write(u'left_labels\t%s\n' % ','.join(self.moves.left_labels))
            #cfg.write(u'right_labels\t%s\n' % ','.join(self.moves.right_labels))
            cfg.write(u'beam_width\t%d\n' % self.beam_width)
            cfg.write(u'auto_pos\t%s\n' % self.auto_pos)
            #if not self.features.ngrams:
            #    cfg.write(u'ngrams\t-1\n')
            #else:
            #    ngram_strs = ['_'.join([str(i) for i in ngram])
            #                  for ngram in self.features.ngrams]
            #    cfg.write(u'ngrams\t%s\n' % u','.join(ngram_strs))
            cfg.write(u'feat_set\t%s\n' % self.feat_set)


def _parse_labels_str(labels_str):
    return [index.hashes.encode_label(l) for l in labels_str.split(',')]
