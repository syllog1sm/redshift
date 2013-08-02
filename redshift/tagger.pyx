from redshift._state cimport *
from redshift.beam cimport TaggerBeam, TagState, TagKernel
from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
import index.hashes
cimport index.hashes
from redshift.io_parse cimport Sentences, Sentence
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy, memset
from libc.stdint cimport uint64_t, int64_t

from os.path import join as pjoin
import os
import os.path
from os.path import join as pjoin
import random
import shutil

DEBUG = False

cdef class BeamTagger:
    cdef Extractor features
    cdef Perceptron guide
    cdef object model_dir
    cdef size_t beam_width
    cdef int feat_thresh
    cdef size_t nr_tag
    cdef size_t _oracle_beam

    cdef size_t* _context
    cdef uint64_t* _features

    def __cinit__(self, model_dir, feat_set="basic", feat_thresh=5, beam_width=4,
                  clean=False):
        self.model_dir = model_dir
        if clean and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            os.mkdir(model_dir)
            self.new_idx(model_dir)
        else:
            self.load_idx(model_dir)
            self.guide.load(pjoin(model_dir, 'tagger.gz'), thresh=self.feat_thresh)
        self.feat_thresh = feat_thresh
        self.guide = Perceptron(100, pjoin(model_dir, 'tagger.gz'))
        if not clean:
            self.guide.load(pjoin(model_dir, 'tagger.gz'), thresh=self.feat_thresh)
        #self.features = Extractor(unigrams + bigrams + trigrams, [])
        self.features = Extractor(basic + clusters, [], bag_of_words=[P1w, P2w,
                                                                      P3w, P4w,
                                                                      P5w, P6w,
                                                                      P7w])
        self.nr_tag = 0
        self.beam_width = beam_width
        self._context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        max_feats = self.features.nr_template + self.features.nr_bow + 2
        self._features = <uint64_t*>calloc(max_feats, sizeof(uint64_t))

    def add_tags(self, Sentences sents):
        cdef size_t i
        n = 0
        self._oracle_beam = 0
        for i in range(sents.length):
            self.tag(sents.s[i])
            n += (sents.s[i].length - 2)
        print '%.4f' % (float(self._oracle_beam) / n), self._oracle_beam, n

    cdef int tag(self, Sentence* sent) except -1:
        cdef TaggerBeam beam = TaggerBeam(None, self.beam_width, sent.length, self.nr_tag)
        cdef size_t p_idx
        cdef size_t kernel
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        self.guide.cache.flush()
        for i in range(sent.length - 1):
            for p_idx in range(beam.bsize):
                pred = <TagState*>beam.beam[p_idx]
                fill_tag_kernel(pred, i)
                beam_scores[p_idx] = self._predict(sent, &pred.kernel)
            beam.extend_states(beam_scores)
        self._oracle_beam += beam.eval_beam(sent.pos)
        beam.fill_parse(sent.parse.moves, sent.pos, sent.parse.heads,
                        sent.parse.labels, sent.parse.sbd, sent.parse.edits)
        # TODO: dealloc tag beam
        free(beam_scores)

    def train(self, Sentences sents, nr_iter=10):
        indices = list(range(sents.length))
        self.nr_tag = 0
        tags = set()
        for i in range(sents.length):
            for j in range(sents.s[i].length):
                if sents.s[i].pos[j] >= self.nr_tag:
                    self.nr_tag = sents.s[i].pos[j]
                    tags.add(sents.s[i].pos[j])
        self.nr_tag += 1
        self.guide.set_classes(range(self.nr_tag))
        if not DEBUG:
            # Extra trick: sort by sentence length for first iteration
            indices.sort(key=lambda i: sents.s[i].length)
        for n in range(nr_iter):
            for i in indices:
                if DEBUG:
                    print ' '.join(sents.strings[i][0])
                self.static_train(n, sents.s[i])
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

    cdef int static_train(self, int iter_num, Sentence* sent) except -1:
        cdef size_t  i
        cdef double* scores
        cdef TaggerBeam beam = TaggerBeam(None, self.beam_width, sent.length, self.nr_tag)
        cdef TagState* gold = <TagState*>malloc(sizeof(TagState))
        gold.tags = <size_t*>calloc(sent.length, sizeof(size_t))
        gold.score = 0
        cdef TagState* pred
        cdef TagState* violn = <TagState*>malloc(sizeof(TagState))
        violn.tags = <size_t*>calloc(sent.length, sizeof(size_t))
        violn.kernel.i = 0
        violn.score = 0
        cdef double** beam_scores = <double**>malloc(beam.k * sizeof(double*))
        self.guide.cache.flush()
        for word_i in range(sent.length - 1):
            for i in range(beam.bsize):
                pred = <TagState*>beam.beam[i]
                fill_tag_kernel(pred, word_i)
                beam_scores[i] = self._predict(sent, &pred.kernel)
            beam.extend_states(beam_scores)
            fill_tag_kernel(gold, word_i)
            scores = self._predict(sent, &gold.kernel)
            gold.score += scores[sent.pos[word_i]]
            gold.tags[word_i] = sent.pos[word_i]
            pred = <TagState*>beam.beam[0]
            if (pred.score - gold.score) >= violn.score:
                violn.score = pred.score - gold.score
                violn.kernel.i = word_i + 1
                memcpy(violn.tags, pred.tags, (word_i + 1) * sizeof(size_t))
        if violn.kernel.i == 0:
            self.guide.n_corr += beam.t
            self.guide.total += beam.t
        else:
            fill_tag_kernel(violn, violn.kernel.i)
            counts = self._count_feats(sent, violn.kernel.i, violn, gold)
            self.guide.batch_update(counts)
        free(violn.tags)
        free(gold.tags)
        free(violn)
        free(gold)
        free(beam_scores)

    cdef double* _predict(self, Sentence* sent, TagKernel* kernel) except NULL:
        cdef bint cache_hit = False
        cdef double* scores
        scores = self.guide.cache.lookup(sizeof(TagKernel), kernel, &cache_hit)
        if not cache_hit:
            #print sent.length, kernel.i, kernel.ptag, kernel.pptag, kernel.ppptag
            fill_context(self._context, sent, kernel)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
        return scores

    cdef dict _count_feats(self, Sentence* sent, size_t t,
                           TagState* pred, TagState* gold):
        cdef size_t d, i, f
        cdef uint64_t* feats
        # Find where the states diverge
        cdef dict counts = {}
        for clas in range(self.nr_tag):
            counts[clas] = {}
        cdef bint seen_diff = False
        for i in range(1, t):
            self.guide.total += 1
            if not seen_diff and gold.tags[i] == pred.tags[i]:
                self.guide.n_corr += 1
                continue
            if gold.tags[i] == pred.tags[i]:
                self.guide.n_corr += 1
            seen_diff = True
            fill_tag_kernel(gold, i)
            fill_context(self._context, sent, &gold.kernel)
            self.features.extract(self._features, self._context)
            self._inc_feats(counts[gold.tags[i]], self._features, 1.0)
            fill_tag_kernel(pred, i)
            fill_context(self._context, sent, &pred.kernel)
            self.features.extract(self._features, self._context)
            self._inc_feats(counts[pred.tags[i]], self._features, -1.0)
        return counts

    cdef int _inc_feats(self, dict counts, uint64_t* feats,
                        double inc) except -1:
        cdef size_t f = 0
        while feats[f] != 0:
            if feats[f] not in counts:
                counts[feats[f]] = 0
            counts[feats[f]] += inc
            f += 1

    def save(self):
        self.guide.save(pjoin(self.model_dir, 'tagger.tgz'))

    def load(self):
        self.guide.load(pjoin(self.model_dir, 'model.gz'), thresh=self.feat_thresh)

    def new_idx(self, model_dir):
        index.hashes.init_word_idx(pjoin(model_dir, 'words'))
        index.hashes.init_pos_idx(pjoin(model_dir, 'pos'))
        index.hashes.init_label_idx(pjoin(model_dir, 'labels'))

    def load_idx(self, model_dir):
        index.hashes.load_word_idx(pjoin(model_dir, 'words'))
        index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        index.hashes.load_label_idx(pjoin(model_dir, 'labels'))
 

cdef inline int fill_tag_kernel(TagState* s, size_t i):
    s.kernel.i = i
    if i >= 1:
        s.kernel.ptag = s.tags[i - 1]
        if i >= 2:
            s.kernel.pptag = s.tags[i - 2]
            if i >= 3:
                s.kernel.ppptag = s.tags[i - 3]
            else:
                s.kernel.ppptag = 0
        else:
            s.kernel.pptag = 0
            s.kernel.ppptag = 0
    else:
        s.kernel.ptag = 0
        s.kernel.pptag = 0
        s.kernel.ppptag = 0


def print_train_msg(n, n_corr, n_move, n_hit, n_miss):
    pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
    move_acc = pc(n_corr, n_move)
    cache_use = pc(n_hit, n_hit + n_miss + 1e-100)
    msg = "#%d: Moves %d/%d=%s" % (n, n_corr, n_move, move_acc)
    if cache_use != 0:
        msg += '. Cache use %s' % cache_use
    print msg


cdef enum:
    N0w
    N0c
    N0c6
    N0c4
    N0suff
    N0pre

    N1w
    N1c
    N1c6
    N1c4
    N1suff
    N1pre

    P1w
    P1c
    P1c6
    P1c4
    P1suff
    P1pre
    P1p

    P2w
    P2c
    P2c6
    P2c4
    P2suff
    P2pre
    P2p

    P3w # For BOW
    P4w
    P5w
    P6w
    P7w
    CONTEXT_SIZE


basic = (
    (N0w,),
    (N1w,),
    (P1p,),
    (P2p,),
    (P1p, P2p),
    (P1p, N0w),
    (N0suff,),
    (N0pre,),
    (N1suff,),
    (N1pre,),
    (P1suff,),
    (P1pre,),
)

clusters = (
    (N0c,),
    (N0c4,),
    (N0c6,),
    (P1c,),
    (P1c4,),
    (P1c6,),
    (N1c,),
    (N1c4,),
    (N1c6,),
    (P1c, N0w),
    (P1p, P1c6, N0w),
    (P1c6, N0w),
    (N0w, N1c),
    (N0w, N1c6),
    (N0w, N1c4),
    (P2c4, P1c4, N0w)
)

cdef int fill_context(size_t* context, Sentence* sent, TagKernel* k):
    cdef size_t i
    for i in range(CONTEXT_SIZE):
        context[i] = 0
    #memset(context, 0, sizeof(size_t) * CONTEXT_SIZE)
    context[N0w] = sent.words[k.i]
    context[N0c] = sent.clusters[k.i]
    context[N0c6] = sent.cprefix6s[k.i]
    context[N0c4] = sent.cprefix4s[k.i]
    context[N0suff] = sent.orths[k.i]
    context[N0pre] = sent.parens[k.i]
    
    context[N1w] = sent.words[k.i+1]
    context[N1c] = sent.clusters[k.i+1]
    context[N1c6] = sent.cprefix6s[k.i+1]
    context[N1c4] = sent.cprefix4s[k.i+1]
    context[N1suff] = sent.orths[k.i + 1]
    context[N1pre] = sent.parens[k.i + 1]
    if k.i == 1:
        return 0
    context[P1w] = sent.words[k.i-1]
    context[P1c] = sent.clusters[k.i-1]
    context[P1c6] = sent.cprefix6s[k.i-1]
    context[P1c4] = sent.cprefix4s[k.i-1]
    context[P1suff] = sent.orths[k.i-1]
    context[P1pre] = sent.parens[k.i-1]
    context[P1p] = k.ptag
    
    if k.i == 2:
        return 0
    context[P2w] = sent.words[k.i-2]
    context[P2c] = sent.clusters[k.i-2]
    context[P2c6] = sent.cprefix6s[k.i-2]
    context[P2c4] = sent.cprefix4s[k.i-2]
    context[P2suff] = sent.orths[k.i-2]
    context[P2pre] = sent.parens[k.i-2]
    context[P2p] = k.pptag

    # Fill bag-of-words slots
    i = k.i - 2
    cdef size_t slot = P3w
    while i > 0 and slot < CONTEXT_SIZE:
        i -= 1
        context[slot] = sent.words[i]
        slot += 1
