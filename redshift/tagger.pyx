from redshift._state cimport *
from redshift.beam cimport TaggerBeam, TagState, fill_hist
#, TagKernel
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
    cdef double** beam_scores

    def __cinit__(self, model_dir, feat_set="basic", feat_thresh=5, beam_width=8,
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
        self.nr_tag = 100
        self.beam_width = beam_width
        self._context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        max_feats = self.features.nr_template + self.features.nr_bow + 100
        self._features = <uint64_t*>calloc(max_feats, sizeof(uint64_t))
        self.beam_scores = <double**>malloc(sizeof(double*) * self.beam_width)
        for i in range(self.beam_width):
            self.beam_scores[i] = <double*>calloc(self.nr_tag, sizeof(double))

    def add_tags(self, Sentences sents):
        cdef size_t i
        n = 0
        for i in range(sents.length):
            self.tag(sents.s[i])
            n += (sents.s[i].length - 2)

    cdef int tag(self, Sentence* sent) except -1:
        cdef TaggerBeam beam = TaggerBeam(None, self.beam_width, sent.length, self.nr_tag)
        cdef size_t p_idx
        cdef TagState* s
        for i in range(sent.length - 1):
            for p_idx in range(beam.bsize):
                s = <TagState*>beam.beam[p_idx]
                fill_context(self._context, sent, s.hist[0], s.hist[1], i)
                self.features.extract(self._features, self._context)
                self.guide.fill_scores(self._features, self.beam_scores[p_idx])
            beam.extend_states(self.beam_scores)
        s = <TagState*>beam.beam[0]
        fill_hist(sent.pos, s, sent.length - 2)

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
        cdef size_t  i, word_i
        cdef TagState* s
        cdef double* gscores = <double*>calloc(self.nr_tag, sizeof(double))
        cdef TaggerBeam beam = TaggerBeam(None, self.beam_width, sent.length, self.nr_tag)
        cdef TagState* violn = <TagState*>calloc(1, sizeof(TagState))
        cdef double gscore = 0
        cdef size_t t = 0
        for word_i in range(sent.length - 1):
            for i in range(beam.bsize):
                s = <TagState*>beam.beam[i]
                # Here s.hist[0] is the prev tag, and s.hist[1] the prev prev tag
                fill_context(self._context, sent, s.hist[0], s.hist[1], word_i)
                self.features.extract(self._features, self._context)
                self.guide.fill_scores(self._features, self.beam_scores[i])
            # After extend_states, beam has the tag for word_i at hist[0] and
            # the tag for the prev word at hist[1]
            beam.extend_states(self.beam_scores)
            fill_context(self._context, sent, sent.pos[word_i-1] if word_i >= 1 else 0,
                         sent.pos[word_i-2] if word_i >= 2 else 0, word_i)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, gscores)
            gscore += gscores[sent.pos[word_i]]
            s = <TagState*>beam.beam[0]
            if (s.score - gscore) >= violn.score:
                violn.score = s.score - gscore
                violn.prev = s.prev
                # The tag for word_i
                violn.hist[0] = s.hist[0]
                # The tag for word_i-1
                violn.hist[1] = s.hist[1]
                t = word_i
            self.guide.n_corr += (sent.pos[word_i] == s.hist[0])
            self.guide.total += 1
        cdef size_t* phist
        if t != 0:
            phist = <size_t*>calloc(t + 1, sizeof(size_t))
            fill_hist(phist, violn, t)
            counts = self._count_feats(sent, t, phist, sent.pos)
            self.guide.batch_update(counts)
        free(violn)
        free(gscores)
        if t != 0:
            free(phist)

    cdef dict _count_feats(self, Sentence* sent, size_t t, size_t* phist,
                           size_t* ghist):
        cdef size_t d, i, f
        cdef uint64_t* feats
        # Find where the states diverge
        cdef dict counts = {}
        for clas in range(self.nr_tag):
            counts[clas] = {}
        for i in range(1, t+1):
            gclas = ghist[i]
            pclas = phist[i]
            gprev = ghist[i-1]
            pprev = phist[i-1]
            gpprev = ghist[i-2] if i >= 2 else 0
            ppprev = phist[i-2] if i >= 2 else 0
            if gclas == pclas:
                if gpprev == pprev and gpprev == ppprev:
                    continue
            fill_context(self._context, sent, gprev, gpprev, i)
            self.features.extract(self._features, self._context)
            self._inc_feats(counts[gclas], self._features, 1.0)
            fill_context(self._context, sent, pprev, ppprev, i)
            self.features.extract(self._features, self._context)
            self._inc_feats(counts[pclas], self._features, -1.0)
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

    N0quo

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
    (N0quo,),
    (N0w, N0quo),
    (P1p, N0quo),
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

cdef int fill_context(size_t* context, Sentence* sent, size_t ptag, size_t pptag, size_t i):
    context[N0w] = sent.words[i]
    context[N0c] = sent.clusters[i]
    context[N0c6] = sent.cprefix6s[i]
    context[N0c4] = sent.cprefix4s[i]
    context[N0suff] = sent.orths[i]
    context[N0pre] = sent.parens[i]
    
    context[N1w] = sent.words[i + 1]
    context[N1c] = sent.clusters[i + 1]
    context[N1c6] = sent.cprefix6s[i + 1]
    context[N1c4] = sent.cprefix4s[i + 1]
    context[N1suff] = sent.orths[i + 1]
    context[N1pre] = sent.parens[i + 1]

    context[N0quo] = sent.quotes[i] == 0
    if i == 1:
        return 0
    context[P1w] = sent.words[i-1]
    context[P1c] = sent.clusters[i-1]
    context[P1c6] = sent.cprefix6s[i-1]
    context[P1c4] = sent.cprefix4s[i-1]
    context[P1suff] = sent.orths[i-1]
    context[P1pre] = sent.parens[i-1]
    context[P1p] = ptag
    if i == 2:
        return 0
    context[P2w] = sent.words[i-2]
    context[P2c] = sent.clusters[i-2]
    context[P2c6] = sent.cprefix6s[i-2]
    context[P2c4] = sent.cprefix4s[i-2]
    context[P2suff] = sent.orths[i-2]
    context[P2pre] = sent.parens[i-2]
    context[P2p] = pptag

    # Fill bag-of-words slots
    cdef size_t slot = P3w
    i -= 2
    while i > 0 and slot < CONTEXT_SIZE:
        i -= 1
        context[slot] = sent.words[i]
        slot += 1
