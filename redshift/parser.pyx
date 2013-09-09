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

from _fast_state cimport *
#from _state cimport hash_kernel
from io_parse cimport Sentence, Sentences
from io_parse import read_conll, read_pos
from transitions cimport TransitionSystem 
from beam cimport FastBeam
from tagger cimport GreedyTagger, BeamTagger

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


def load_parser(model_dir):
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
    for ngram_str in params.get('ngrams', '-1').split(','):
        if ngram_str == '-1': continue
        ngrams.append(tuple([int(i) for i in ngram_str.split('_')]))
    auto_pos = params['auto_pos'] == 'True'
    params = {'clean': False, 'train_alg': train_alg,
              'feat_set': feat_set, 'feat_thresh': feat_thresh,
              'vocab_thresh': 1, 'allow_reattach': allow_reattach,
              'use_edit': use_edit, 'allow_reduce': allow_reduce,
              'beam_width': beam_width,
              'ngrams': ngrams,
              'auto_pos': auto_pos}
    if beam_width >= 2:
        parser = BeamParser(model_dir, **params)
    else:
        parser = GreedyParser(model_dir, **params)
    pos_tags = set([int(line.split()[0]) for line in open(pjoin(model_dir, 'pos'))])
    parser.load()
    _, nr_label = parser.moves.set_labels(pos_tags, _parse_labels_str(l_labels),
                            _parse_labels_str(r_labels))
    
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
                  allow_reattach=False, allow_reduce=False, use_edit=False,
                  beam_width=1, ngrams=None, auto_pos=False):
        self.model_dir = self.setup_model_dir(model_dir, clean)
        self.feat_set = feat_set
        self.ngrams = ngrams if ngrams is not None else []
        if self.feat_set != 'debug':
            templates = _parse_features.baseline_templates()
        else:
            templates = _parse_features.debug
        #templates += _parse_features.ngram_feats(self.ngrams)
        if 'disfl' in self.feat_set:
            templates += _parse_features.disfl
            templates += _parse_features.new_disfl
            templates += _parse_features.suffix_disfl
        if 'xlabels' in self.feat_set:
            templates += _parse_features.extra_labels
        if 'stack' in self.feat_set:
            templates += _parse_features.stack_second
        if 'hist' in self.feat_set:
            templates += _parse_features.history
        if 'clusters' in self.feat_set:
            templates += _parse_features.clusters
        if 'bitags' in self.feat_set:
            templates += _parse_features.pos_bigrams()
        if 'edges' in self.feat_set:
            templates += _parse_features.edges
        if 'match' in self.feat_set:
            match_feats = _parse_features.match_templates()
            print "Using %d match feats" % len(match_feats)
        else:
            match_feats = []
        self.extractor = Extractor(templates, match_feats)
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(_parse_features.context_size(), sizeof(size_t))
        self.feat_thresh = feat_thresh
        self.train_alg = train_alg
        self.beam_width = beam_width
        self.moves = TransitionSystem(allow_reattach=allow_reattach,
                                      allow_reduce=allow_reduce, use_edit=use_edit)
        self.auto_pos = auto_pos
        self.guide = Perceptron(self.moves.max_class, pjoin(model_dir, 'model.gz'))
        self.tagger = BeamTagger(model_dir, clean=False)

    def setup_model_dir(self, loc, clean):
        if clean and os.path.exists(loc):
            shutil.rmtree(loc)
        if os.path.exists(loc):
            assert os.path.isdir(loc)
        else:
            os.mkdir(loc)
        return loc

    def train(self, str train_str, unlabelled=False, n_iter=15):
        cdef size_t i, j, n
        cdef Sentence* sent
        cdef Sentences held_out_gold
        cdef Sentences held_out_parse

        cdef Sentences sents = read_conll(train_str,
                                          unlabelled=unlabelled)
        self.say_config()
        self.tagger.setup_classes(sents)
        move_classes, nr_label = self.moves.set_labels(*sents.get_labels())
        #self.features.set_nr_label(nr_label)
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
                if self.auto_pos:
                    self.tagger.train_sent(sents.s[i])
                if self.train_alg == 'static':
                    self.static_train(n, sents.s[i])
                else:
                    self.dyn_train(n, sents.s[i])
            if self.auto_pos:
                print_train_msg(n, self.tagger.guide.n_corr, self.tagger.guide.total,
                                0, 0)
            print_train_msg(n, self.guide.n_corr, self.guide.total, self.guide.cache.n_hit,
                            self.guide.cache.n_miss)
            if self.guide.n_corr == self.guide.total:
                break
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
    
    def parse_file(self, in_loc, out_loc):
        cdef Sentences sents = read_pos(open(in_loc).read())
        self.guide.nr_class = self.moves.nr_class
        cdef size_t i
        for i in range(sents.length):
            self.parse(sents.s[i])
        sents.write_parses(open(out_loc, 'w'))

    cdef int parse(self, Sentence* sent) except -1:
        raise NotImplementedError

    def save(self):
        self.guide.save(pjoin(self.model_dir, 'model.gz'))
        self.tagger.save()
        index.hashes.save_idx('word', pjoin(self.model_dir, 'words'))
        index.hashes.save_idx('pos', pjoin(self.model_dir, 'pos'))
        index.hashes.save_idx('labels', pjoin(self.model_dir, 'labels'))
   
    def load(self):
        self.guide.load(pjoin(self.model_dir, 'model.gz'), thresh=self.feat_thresh)
        self.tagger.guide.load(pjoin(self.model_dir, 'tagger.gz'), thresh=self.feat_thresh)
        index.hashes.load_idx('word', pjoin(self.model_dir, 'words'))
        index.hashes.load_idx('pos', pjoin(self.model_dir, 'pos'))
        index.hashes.load_idx('label', pjoin(self.model_dir, 'labels'))
   
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
            cfg.write(u'auto_pos\t%s\n' % self.auto_pos)
            cfg.write(u'feat_set\t%s\n' % self.feat_set)

    def __dealloc__(self):
        pass


cdef class BeamParser(BaseParser):
    cdef int parse(self, Sentence* sent) except -1:
        cdef FastBeam beam = FastBeam(self.moves, self.beam_width, sent.length)
        cdef size_t p_idx
        cdef Kernel* kernel
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        if self.auto_pos:
            self.tagger.tag(sent)
        self.guide.cache.flush()
        while not beam.is_finished:
            for p_idx in range(beam.bsize):
                pred = beam.beam[p_idx]
                self.moves.fill_valid(beam.valid[p_idx], can_push(&pred.knl, sent.length),
                                      has_stack(&pred.knl), has_head(&pred.knl))
                beam_scores[p_idx] = self._predict(sent, &pred.knl)
            beam.extend_states(beam_scores)
        s = <FastState*>beam.beam[0]
        sent.parse.n_moves = beam.t
        beam.fill_parse(sent.parse.moves, sent.pos, sent.parse.heads, sent.parse.labels,
                        sent.parse.sbd, sent.parse.edits)
        free(beam_scores)

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t  i
        cdef Kernel* kernel
        cdef double* scores
        cdef FastBeam beam = FastBeam(self.moves, self.beam_width, sent.length)
        cdef FastState* gold = init_fast_state()
        cdef FastState* pred
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        cdef double max_violn = 0
        cdef FastState* upd_g
        cdef FastState* upd_p
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
                pred = beam.beam[i]
                self.moves.fill_valid(beam.valid[i], can_push(&pred.knl, sent.length),
                                      has_stack(&pred.knl), has_head(&pred.knl))
                beam_scores[i] = self._predict(sent, &pred.knl)
            beam.extend_states(beam_scores)

            oracle = self.moves.break_tie(can_push(&gold.knl, sent.length),
                                          has_head(&gold.knl),
                                          gold.knl.i, gold.knl.s0, sent.length, sent.pos,
                                          sent.parse.heads, sent.parse.labels,
                                          sent.parse.edits)
            scores = self._predict(sent, &gold.knl)
            pred = beam.beam[0]
            gold = extend_fstate(gold, self.moves.moves[oracle], self.moves.labels[oracle],
                                 oracle, scores[oracle], 0)
            if pred.clas == gold.clas:
                self.guide.n_corr += 1
            self.guide.total += 1
            delta = pred.score - gold.score
            if delta >= max_violn:
                max_violn = delta
                upd_g = gold
                upd_p = pred
        if upd_g != NULL:
            counted = self._count_feats(sent, upd_g, upd_p)
            if counted:
                self.guide.batch_update(counted)
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags)
        free(beam_scores)
        cdef FastState* prev = gold.prev
        while gold != NULL:
            prev = gold.prev
            free(gold)
            gold = prev

    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t i
        cdef Kernel* kernel
        cdef FastState* p
        cdef FastState* g
        cdef FastState* upd_p
        cdef FastState* upd_g
        cdef size_t* bu_tags
        cdef size_t stack_len
        cdef size_t* stack = <size_t*>calloc(sent.length, sizeof(size_t))
        if self.auto_pos:
            bu_tags = <size_t*>calloc(sent.length, sizeof(size_t))
            memcpy(bu_tags, sent.pos, sent.length * sizeof(size_t))
            self.tagger.tag(sent)
        pred = FastBeam(self.moves, self.beam_width, sent.length)
        gold = FastBeam(self.moves, self.beam_width, sent.length)
        pred_scores = <double**>malloc(self.beam_width * sizeof(double*))
        gold_scores = <double**>malloc(self.beam_width * sizeof(double*))
        cdef double max_violn = 0
        self.guide.cache.flush()
        while not pred.is_finished and not gold.is_finished:
            for i in range(pred.bsize):
                p = pred.beam[i]
                self.moves.fill_valid(pred.valid[i], can_push(&p.knl, sent.length),
                                      has_stack(&p.knl), has_head(&p.knl))
                pred_scores[i] = self._predict(sent, &p.knl)
                stack_len = fill_stack(stack, p)
                self.moves.fill_costs(pred.costs[i], p.knl.i, sent.length,
                                     stack_len, stack, p.knl.Ls0 != 0, sent.pos,
                                     sent.parse.heads, sent.parse.labels, sent.parse.edits)
            pred.extend_states(pred_scores)
            for i in range(gold.bsize):
                g = gold.beam[i]
                self.moves.fill_valid(gold.valid[i], can_push(&g.knl, sent.length),
                                      has_stack(&p.knl), has_head(&p.knl))
                gold_scores[i] = self._predict(sent, &g.knl)
                stack_len = fill_stack(stack, g)
                self.moves.fill_costs(gold.costs[i], g.knl.i, sent.length, stack_len,
                                      stack, g.knl.Ls0 != 0, sent.pos, 
                                      sent.parse.heads, sent.parse.labels,
                                      sent.parse.edits)
                for clas in range(self.moves.nr_class):
                    if gold.costs[i][clas] != 0:
                        gold.valid[i][clas] = -1
            gold.extend_states(gold_scores)
            g = gold.beam[0]
            p = pred.beam[0]
            if p.score - g.score >= max_violn and p.cost >= 1:
                max_violn = p.score - g.score
                upd_g = g
                upd_p = p
            self.guide.n_corr += p.clas == g.clas
            self.guide.total += 1
        if max_violn >= 0:
            counted = self._count_feats(sent, upd_g, upd_p)
            self.guide.batch_update(counted)
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags)
        free(pred_scores)
        free(gold_scores)
        free(stack)

    def say_config(self):
        beam_settings = (self.beam_width, self.train_alg)
        print 'Beam settings: k=%d; upd_strat=%s' % beam_settings

    cdef double* _predict(self, Sentence* sent, Kernel* kernel) except NULL:
        cdef bint cache_hit = False
        scores = self.guide.cache.lookup(sizeof(Kernel), kernel, &cache_hit)
        if not cache_hit:
            fill_context(self._context, self.moves.n_labels, sent.words,
                         sent.pos, sent.clusters, sent.cprefix6s, sent.cprefix4s,
                         sent.prefix, sent.parens, sent.quotes, kernel,
                         &kernel.s0l, &kernel.s0r, &kernel.n0l)
            self.extractor.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
        return scores

    cdef dict _count_feats(self, Sentence* sent, FastState* g, FastState* p):
        cdef dict counts = {}
        for clas in range(self.moves.nr_class):
            counts[clas] = {}
        cdef bint seen_diff = False
        while g.prev != NULL and p.prev != NULL:
            if g.clas != p.clas:
                seen_diff = True
            if seen_diff and hash_kernel(&g.knl) != hash_kernel(&p.knl):
                self._inc_feats(counts[g.clas], sent, &g.prev.knl, 1.0)
                self._inc_feats(counts[p.clas], sent, &p.prev.knl, -1.0)
            g = g.prev
            p = p.prev
        return counts 

    cdef int _inc_feats(self, dict counts, Sentence* sent, Kernel* k, double inc) except -1:
        fill_context(self._context, self.moves.n_labels, sent.words,
                     sent.pos, sent.clusters, sent.cprefix6s, sent.cprefix4s,
                     sent.prefix, sent.parens, sent.quotes, k,
                     &k.s0l, &k.s0r, &k.n0l)
        self.extractor.extract(self._features, self._context)
 
        cdef size_t f = 0
        while self._features[f] != 0:
            if self._features[f] not in counts:
                counts[self._features[f]] = 0
            counts[self._features[f]] += inc
            f += 1

cdef double STRAY_PC = 0.90
cdef class GreedyParser(BaseParser):
    cdef int parse(self, Sentence* sent) except -1:
        cdef uint64_t* feats
        cdef FastState* s = init_fast_state()
        cdef size_t _ = 0
        sent.parse.n_moves = 0
        if self.auto_pos:
            self.tagger.tag(sent)
        cdef size_t t = 0
        while not is_finished(&s.knl, sent.length):
            feats = self._extract(sent, &s.knl)
            self.moves.fill_valid(self.moves._costs, can_push(&s.knl, sent.length),
                                  has_stack(&s.knl),  has_head(&s.knl))
            # TODO: Fix label guessing
            clas = self._predict(feats, self.moves._costs, &_)
            s = extend_fstate(s, self.moves.moves[clas], self.moves.labels[clas],
                              clas, 0, 0)
            # Take the last set head, to support non-monotonicity
            # Take the heads from states just after right and left arcs
            if s.knl.Ls0 != 0:
                sent.parse.heads[s.knl.s0] = s.knl.s1
                sent.parse.labels[s.knl.s0] = s.knl.Ls0
            if s.knl.n0l.idx[0] != 0:
                sent.parse.heads[s.knl.n0l.idx[0]] = s.knl.i
                sent.parse.labels[s.knl.n0l.idx[0]] = s.knl.n0l.lab[0]
            sent.parse.moves[t] = clas 
        sent.parse.n_moves = t
        free_fstate(s)
        #_state.fill_edits(s, sent.parse.edits)
 
    cdef int dyn_train(self, int iter_num, Sentence* sent) except -1:
        cdef int* valid = <int*>calloc(self.guide.nr_class, sizeof(int))
        cdef size_t pred, stack_len
        cdef uint64_t* feats
        cdef size_t _ = 0
        cdef int* costs = self.moves._costs

        cdef size_t* bu_tags 
        cdef FastState* s = init_fast_state()
        if self.auto_pos:
            bu_tags = <size_t*>calloc(sent.length, sizeof(size_t))
            memcpy(bu_tags, sent.pos, sent.length * sizeof(size_t))
            self.tagger.tag(sent)
        seen_states = set()
        cdef size_t* stack = <size_t*>calloc(sent.length, sizeof(size_t))
        while not is_finished(&s.knl, sent.length):
            self.moves.fill_valid(valid, can_push(&s.knl, sent.length),
                                  has_stack(&s.knl), has_head(&s.knl))
            feats = self._extract(sent, &s.knl)
            # TODO: Fix label guessing here
            pred = self._predict(feats, valid, &_)
            stack_len = fill_stack(stack, s)
            self.moves.fill_costs(costs, s.knl.i, sent.length, stack_len, stack,
                                  has_head(&s.knl), sent.pos, sent.parse.heads,
                                  sent.parse.labels, sent.parse.edits)
            gold = pred if costs[pred] == 0 else self._predict(feats, self.moves._costs, &_)
            self.guide.update(pred, gold, feats, 1)
            clas = pred if iter_num >= 2 and random.random() < STRAY_PC else gold
            seen_states.add(<size_t>s)
            s = extend_fstate(s, self.moves.moves[clas], self.moves.labels[clas],
                              clas, self.guide.scores[clas], costs[clas])
            self.guide.n_corr += (gold == pred)
            self.guide.total += 1
        if self.auto_pos:
            memcpy(sent.pos, bu_tags, sent.length * sizeof(size_t))
            free(bu_tags)
        cdef size_t addr
        for addr in seen_states:
            s = <FastState*>addr
            s = s.prev
            free(<FastState*>addr)
        free(valid)

    def say_config(self):
        if self.moves.allow_reattach and self.moves.allow_reduce:
            print 'NM L+D'
        elif self.moves.allow_reattach:
            print 'NM L'
        elif self.moves.allow_reduce:
            print 'NM D'

    cdef uint64_t* _extract(self, Sentence* sent, Kernel* kernel):
        fill_context(self._context, self.moves.n_labels, sent.words,
                     sent.pos, sent.clusters, sent.cprefix6s, sent.cprefix4s,
                     sent.prefix, sent.parens, sent.quotes, kernel,
                     &kernel.s0l, &kernel.s0r, &kernel.n0l)
        self.extractor.extract(self._features, self._context)
        return self._features
   
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
        self.guide.fill_scores(self._features, scores)
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
