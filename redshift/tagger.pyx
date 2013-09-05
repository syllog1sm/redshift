from redshift._state cimport *
from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
import index.hashes
cimport index.hashes
from ext.murmurhash cimport MurmurHash64A
from ext.sparsehash cimport *


from redshift.io_parse cimport Sentences, Sentence
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy, memset
from libc.stdint cimport uint64_t, int64_t
from libcpp.vector cimport vector 
from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair

cimport cython
from os.path import join as pjoin
import os
import os.path
from os.path import join as pjoin
import random
import shutil
from collections import defaultdict

DEBUG = False


cdef class BaseTagger:
    def __cinit__(self, model_dir, feat_set="basic", feat_thresh=5, beam_width=4,
                  clean=False, trained=False):
        assert not (clean and trained)
        self.model_dir = model_dir
        if clean and os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.feat_thresh = feat_thresh
        #if trained:
        #    self.load_idx(model_dir)
        self.guide = Perceptron(100, pjoin(model_dir, 'tagger.gz'))
        if trained:
            self.guide.load(pjoin(model_dir, 'tagger.gz'), thresh=self.feat_thresh)
        self.features = Extractor(basic + clusters + case + orth, [],
                                  bag_of_words=[P1p, P1alt])
        self.tagdict = dense_hash_map[size_t, size_t]()
        self.tagdict.set_empty_key(0)
        self.nr_tag = 100
        self.beam_width = beam_width
        self._context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        self.max_feats = self.features.nr_template + self.features.nr_bow + 100
        self._features = <uint64_t*>calloc(self.max_feats, sizeof(uint64_t))
        self.beam_scores = <double**>malloc(sizeof(double*) * self.beam_width)
        for i in range(self.beam_width):
            self.beam_scores[i] = <double*>calloc(self.nr_tag, sizeof(double))

    def add_tags(self, Sentences sents):
        cdef size_t i
        for i in range(sents.length):
            self.tag(sents.s[i])
 
    def train(self, Sentences sents, nr_iter=10):
        self.setup_classes(sents)
        indices = list(range(sents.length))
        for n in range(nr_iter):
            for i in indices:
                if DEBUG:
                    print ' '.join(sents.strings[i][0])
                self.train_sent(sents.s[i])
            print_train_msg(n, self.guide.n_corr, self.guide.total)
            self.guide.n_corr = 0
            self.guide.total = 0
            if n % 2 == 1 and self.feat_thresh > 1:
                self.guide.prune(self.feat_thresh)
            if n < 3:
                self.guide.reindex()
            random.shuffle(indices)
        self.guide.finalize()

    def setup_classes(self, Sentences sents):
        self.nr_tag = 0
        tags = set()
        tag_freqs = defaultdict(lambda: defaultdict(int))
        for i in range(sents.length):
            for j in range(sents.s[i].length):
                tag_freqs[sents.s[i].words[j]][sents.s[i].pos[j]] += 1
                if sents.s[i].pos[j] >= self.nr_tag:
                    self.nr_tag = sents.s[i].pos[j]
                    tags.add(sents.s[i].pos[j])
        self.nr_tag += 1
        self.guide.set_classes(range(self.nr_tag))
        types = 0
        tokens = 0
        n = 0
        err = 0
        print "Making tagdict"
        for word, freqs in tag_freqs.items():
            total = sum(freqs.values())
            n += total
            if total >= 100:
                mode, tag = max([(freq, tag) for tag, freq in freqs.items()])
                if float(mode) / total >= 0.99:
                    assert tag != 0
                    self.tagdict[word] = tag
                    types += 1
                    tokens += total
                    err += (total - mode)
        print "%d types" % types
        print "%d/%d=%.4f true" % (err, tokens, (1 - (float(err) / tokens)) * 100)
        print "%d/%d=%.4f cov" % (tokens, n, (float(tokens) / n) * 100)
 
    cdef int tag(self, Sentence* s) except -1:
        raise NotImplementedError

    cdef int train_sent(self, Sentence* sent) except -1:
        raise NotImplementedError

    def save(self):
        self.guide.save(pjoin(self.model_dir, 'tagger.gz'))
        index.hashes.save_idx('word', pjoin(self.model_dir, 'words'))
        index.hashes.save_idx('pos', pjoin(self.model_dir, 'pos'))
        index.hashes.save_idx('label', pjoin(self.model_dir, 'labels'))

    def load(self):
        self.guide.load(pjoin(self.model_dir, 'tagger.gz'), thresh=self.feat_thresh)
        self.nr_tag = self.guide.nr_class
        index.hashes.load_idx(pjoin(self.model_dir, 'words'))
        index.hashes.load_idx(pjoin(self.model_dir, 'pos'))
        index.hashes.load_idx(pjoin(self.model_dir, 'labels'))


cdef class GreedyTagger(BaseTagger):
    def add_tags(self, Sentences sents):
        cdef size_t i
        for i in range(sents.length):
            self.tag(sents.s[i])

    cdef int tag(self, Sentence* sent) except -1:
        cdef size_t i, clas, lookup
        cdef double incumbent, runner_up, score
        cdef size_t prev = sent.pos[0]
        cdef size_t alt = sent.pos[0]
        cdef size_t prevprev = 0
        for i in range(1, sent.length - 1):
            lookup = self.tagdict[sent.words[i]]
            if lookup != 0:
                sent.pos[i] = lookup
                sent.alt_pos[i] = 0
                alt = 0
                prevprev = prev
                prev = lookup
                continue 
            sent.pos[i] = 0
            sent.alt_pos[i] = 0
            fill_context(self._context, sent, prev, prevprev, alt, i)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, self.guide.scores)
            incumbent = -10000
            runner_up = -10000
            for clas in range(self.guide.nr_class):
                score = self.guide.scores[clas]
                if score >= incumbent:
                    sent.alt_pos[i] = sent.pos[i]
                    sent.pos[i] = clas
                    runner_up = incumbent
                    incumbent = score
            prevprev = prev
            prev = sent.pos[i]
            alt = sent.alt_pos[i]

    cdef int train_sent(self, Sentence* sent) except -1:
        cdef size_t w, clas, second, pred, prev, prevprev, lookup
        cdef double score, incumbent, runner_up
        cdef double second_score
        prev = sent.pos[0]
        alt = sent.pos[0]
        for w in range(1, sent.length - 1):
            lookup = self.tagdict[sent.words[w]]
            if lookup != 0:
                alt = 0
                prevprev = prev
                prev = lookup
                continue 
            fill_context(self._context, sent, prev, prevprev, alt, w)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, self.guide.scores)
            incumbent = 0
            runner_up = 0
            pred = 0
            second = 0
            for clas in range(self.nr_tag):
                score = self.guide.scores[clas]
                if score >= incumbent:
                    runner_up = incumbent
                    second = pred
                    incumbent = score
                    pred = clas
            if pred != sent.pos[w]:
                self.guide.update(pred, sent.pos[w], self._features, 1.0)
            else:
                self.guide.n_corr += 1
            self.guide.total += 1
            prevprev = prev
            prev = pred
            alt = second


cdef class BeamTagger(BaseTagger):
    cdef int tag(self, Sentence* sent) except -1:
        cdef TaggerBeam beam = TaggerBeam(self.beam_width, sent.length, self.nr_tag)
        cdef size_t p_idx
        cdef TagState* s
        for i in range(sent.length - 1):
            self.fill_beam_scores(beam, sent, i)
            beam.extend_states(self.beam_scores)
        s = <TagState*>beam.beam[0]
        fill_hist(sent.pos, s, sent.length - 1)

    cdef int fill_beam_scores(self, TaggerBeam beam, Sentence* sent,
                              size_t word_i) except -1:
        for i in range(beam.bsize):
            # At this point, beam.clas is the _last_ prediction, not the prediction
            # for this instance
            fill_context(self._context, sent, beam.parents[i].clas,
                         get_p(beam.parents[i]),
                         beam.parents[i].alt, word_i)
            self.features.extract(self._features, self._context)
            self.guide.fill_scores(self._features, self.beam_scores[i])

    cdef int train_sent(self, Sentence* sent) except -1:
        cdef size_t  i, tmp
        cdef TaggerBeam beam = TaggerBeam(self.beam_width, sent.length, self.nr_tag)
        cdef TagState* gold_state = extend_state(NULL, 0, NULL, 0)
        cdef MaxViolnUpd updater = MaxViolnUpd(self.nr_tag)
        for i in range(sent.length - 1):
            gold_state = self.extend_gold(gold_state, sent, i)
            self.fill_beam_scores(beam, sent, i)
            beam.extend_states(self.beam_scores)
            updater.compare(beam.beam[0], gold_state, i)
            self.guide.n_corr += (gold_state.clas == beam.beam[0].clas)
            self.guide.total += 1
        if updater.delta != -1:
            counts = updater.count_feats(self._features, self._context, sent, self.features)
            self.guide.batch_update(counts)
        cdef TagState* prev
        while gold_state != NULL:
            prev = gold_state.prev
            free(gold_state)
            gold_state = prev

    cdef TagState* extend_gold(self, TagState* s, Sentence* sent, size_t i) except NULL:
        if i >= 1:
            assert s.clas == sent.pos[i - 1]
        else:
            assert s.clas == 0
        fill_context(self._context, sent, s.clas, get_p(s), s.alt, i)
        self.features.extract(self._features, self._context)
        self.guide.fill_scores(self._features, self.guide.scores)
        ext = extend_state(s, sent.pos[i], self.guide.scores, self.guide.nr_class)
        return ext


cdef class MaxViolnUpd:
    cdef TagState* pred
    cdef TagState* gold
    cdef Sentence* sent
    cdef double delta
    cdef int length
    cdef size_t nr_class
    cdef size_t tmp
    def __cinit__(self, size_t nr_class):
        self.delta = -1
        self.length = -1
        self.nr_class = nr_class

    cdef int compare(self, TagState* pred, TagState* gold, size_t i):
        delta = pred.score - gold.score
        if delta > self.delta:
            self.delta = delta
            self.pred = pred
            self.gold = gold
            self.length = i 

    cdef dict count_feats(self, uint64_t* feats, size_t* context, Sentence* sent,
                          Extractor extractor):
        if self.length == -1:
            return {}
        cdef TagState* g = self.gold
        cdef TagState* p = self.pred
        cdef int i = self.length
        cdef dict counts = {}
        for clas in range(self.nr_class):
            counts[clas] = {} 
        cdef size_t gclas, gprev, gprevprev
        cdef size_t pclas, pprev, prevprev
        while g != NULL and p != NULL and i >= 0:
            gclas = g.clas
            gprev = get_p(g)
            gprevprev = get_pp(g)
            galt = g.alt
            pclas = p.clas
            pprev = get_p(p)
            pprevprev = get_pp(p)
            palt = p.alt
            if gclas == pclas and pprev == gprev and gprevprev == pprevprev:
                g = g.prev
                p = p.prev
                i -= 1
                continue
            fill_context(context, sent, gprev, gprevprev,
                         g.prev.alt if g.prev != NULL else 0, i)
            extractor.extract(feats, context)
            self._inc_feats(counts[gclas], feats, 1.0)
            fill_context(context, sent, pprev, pprevprev,
                         p.prev.alt if p.prev != NULL else 0, i)
            extractor.extract(feats, context)
            self._inc_feats(counts[p.clas], feats, -1.0)
            assert sent.words[i] == context[N0w]
            g = g.prev
            p = p.prev
            i -= 1
        return counts

    cdef int _inc_feats(self, dict counts, uint64_t* feats,
                        double inc) except -1:
        cdef size_t f = 0
        while feats[f] != 0:
            if feats[f] not in counts:
                counts[feats[f]] = 0
            counts[feats[f]] += inc
            f += 1


def print_train_msg(n, n_corr, n_move):
    pc = lambda a, b: '%.1f' % ((float(a) / (b + 1e-100)) * 100)
    move_acc = pc(n_corr, n_move)
    msg = "#%d: Moves %d/%d=%s" % (n, n_corr, n_move, move_acc)
    print msg


cdef enum:
    N0w
    N0c
    N0c6
    N0c4
    N0suff
    N0pre

    N0quo
    N0paren
    N0title
    N0upper
    N0alpha

    N1title
    N1upper
    N1alpha

    P1title
    P1upper
    P1alpha

    N1w
    N1c
    N1c6
    N1c4
    N1suff
    N1pre

    N2w
    N2c
    N2c6
    N2c4
    N2suff
    N2pre

    P1w
    P1c
    P1c6
    P1c4
    P1suff
    P1pre
    P1p
    P1alt

    P2w
    P2c
    P2c6
    P2c4
    P2suff
    P2pre
    P2p

    N3w
    N4w

    N0_label
    N0_head_w
    N0_head_p
    P1_label
    P1_head_w

    CONTEXT_SIZE


basic = (
    (N0w,),
    (N1w,),
    (P1w,),
    (P2w,),
    (P1p,),
    (P2p,),
    (P1p, P2p),
    (P1p, N0w),
    (N0suff,),
    (N1suff,),
    (P1suff,),
    (N2w,),
    (N3w,),
    (P1p, P1alt),
)

case = (
    (N0title,),
    (N0upper,),
    (N0alpha,),
    (N0title, N0suff),
    (N0title, N0upper, N0alpha),
    (P1title,),
    (P1upper,),
    (P1alpha,),
    (N1title,),
    (N1upper,),
    (N1alpha,),
    (P1title, N0title, N1title),
    (P1p, N0title,),
    (P1p, N0upper,),
    (P1p, N0alpha,),
    (P1title, N0w),
    (P1upper, N0w),
    (P1title, N0w, N1title),
    (N0title, N0upper, N0c),
)

parse = (
    (N0_label,),
    (N0_head_w,),
    (N0_head_p,),
    (P1_head_w,),
    (P1_label,),
    (N0_label, P1_label),
    #(N0_left_w,),
    #(N0_left_p,),
)

orth = (
    (N0pre,),
    (N1pre,),
    (P1pre,),
    (N0quo,),
    (N0w, N0quo),
    (P1p, N0quo),
    (N0w, N0paren),
    (P1p, N0paren)
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
    (N2c,),
    (P1c, N0w),
    (P1p, P1c6, N0w),
    (P1c6, N0w),
    (N0w, N1c),
    (N0w, N1c6),
    (N0w, N1c4),
    (P2c4, P1c4, N0w)
)


cdef int fill_context(size_t* context, Sentence* sent, size_t ptag, size_t pptag,
                      size_t p_alt, size_t i):
    for j in range(CONTEXT_SIZE):
        context[j] = 0
    context[N0w] = sent.words[i]
    context[N0c] = sent.clusters[i]
    context[N0c6] = sent.cprefix6s[i]
    context[N0c4] = sent.cprefix4s[i]
    context[N0suff] = sent.suffix[i]
    context[N0pre] = sent.prefix[i]
    
    context[N1w] = sent.words[i + 1]
    context[N1c] = sent.clusters[i + 1]
    context[N1c6] = sent.cprefix6s[i + 1]
    context[N1c4] = sent.cprefix4s[i + 1]
    context[N1suff] = sent.suffix[i + 1]
    context[N1pre] = sent.prefix[i + 1]

    context[N0quo] = sent.quotes[i] != 0
    context[N0paren] = sent.parens[i] != 0
    context[N0alpha] = sent.non_alpha[i]
    context[N0upper] = sent.oft_upper[i]
    context[N0title] = sent.oft_title[i]

    context[N1alpha] = sent.non_alpha[i+1]
    context[N1upper] = sent.oft_upper[i+1]
    context[N1title] = sent.oft_title[i+1]

    context[N0_label] = sent.parse.labels[i]
    context[N0_head_w] = sent.words[sent.parse.heads[i]]
    context[N0_head_p] = sent.pos[sent.parse.heads[i]]
    if (i + 2) < sent.length:
        context[N2w] = sent.words[i + 2]
        context[N2c] = sent.clusters[i + 2]
        context[N2c6] = sent.cprefix6s[i + 2]
        context[N2c4] = sent.cprefix4s[i + 2]
        context[N2suff] = sent.suffix[i + 2]
        context[N2pre] = sent.prefix[i + 2]
        if (i + 3) < sent.length:
            context[N3w] = sent.words[i + 3]
    if i == 0:
        return 0
    context[P1w] = sent.words[i-1]
    context[P1c] = sent.clusters[i-1]
    context[P1c6] = sent.cprefix6s[i-1]
    context[P1c4] = sent.cprefix4s[i-1]
    context[P1suff] = sent.suffix[i-1]
    context[P1pre] = sent.prefix[i-1]
    context[P1p] = ptag
    context[P1alt] = p_alt

    context[P1upper] = sent.oft_upper[i-1]
    context[P1alpha] = sent.non_alpha[i-1]
    context[P1title] = sent.oft_title[i-1]

    context[P1_label] = sent.parse.labels[i-1]
    context[P1_head_w] = sent.words[sent.parse.heads[i-1]]
    if i == 1:
        return 0
    context[P2w] = sent.words[i-2]
    context[P2c] = sent.clusters[i-2]
    context[P2c6] = sent.cprefix6s[i-2]
    context[P2c4] = sent.cprefix4s[i-2]
    context[P2suff] = sent.suffix[i-2]
    context[P2pre] = sent.prefix[i-2]
    context[P2p] = pptag


cdef class TaggerBeam:
    def __cinit__(self, size_t k, size_t length, nr_tag=None):
        self.nr_class = nr_tag
        self.k = k
        self.t = 0
        self.bsize = 1
        self.is_full = self.bsize >= self.k
        self.seen_states = set()
        self.beam = <TagState**>malloc(k * sizeof(TagState*))
        self.parents = <TagState**>malloc(k * sizeof(TagState*))
        cdef size_t i
        for i in range(k):
            self.parents[i] = extend_state(NULL, 0, NULL, 0)
            self.seen_states.add(<size_t>self.parents[i])

    @cython.cdivision(True)
    cdef int extend_states(self, double** ext_scores) except -1:
        # Former states are now parents, beam will hold the extensions
        cdef size_t i, clas, move_id
        cdef double parent_score, score
        cdef double* scores
        cdef priority_queue[pair[double, size_t]] next_moves
        next_moves = priority_queue[pair[double, size_t]]()
        for i in range(self.bsize):
            scores = ext_scores[i]
            for clas in range(self.nr_class):
                score = self.parents[i].score + scores[clas]
                move_id = (i * self.nr_class) + clas
                next_moves.push(pair[double, size_t](score, move_id))
        cdef pair[double, size_t] data
        # Apply extensions for best continuations
        cdef TagState* s
        cdef TagState* prev
        cdef size_t addr
        cdef dense_hash_map[uint64_t, bint] seen_equivs = dense_hash_map[uint64_t, bint]()
        seen_equivs.set_empty_key(0)
        self.bsize = 0
        while self.bsize < self.k and not next_moves.empty():
            data = next_moves.top()
            i = data.second / self.nr_class
            clas = data.second % self.nr_class
            prev = self.parents[i]
            hashed = (clas * self.nr_class) + prev.clas
            if seen_equivs[hashed]:
                next_moves.pop()
                continue
            seen_equivs[hashed] = 1
            self.beam[self.bsize] = extend_state(prev, clas, ext_scores[i],
                                                 self.nr_class)
            addr = <size_t>self.beam[self.bsize]
            self.seen_states.add(addr)
            next_moves.pop()
            self.bsize += 1
        for i in range(self.bsize):
            self.parents[i] = self.beam[i]
        self.is_full = self.bsize >= self.k
        self.t += 1

    def __dealloc__(self):
        cdef TagState* s
        cdef size_t addr
        for addr in self.seen_states:
            s = <TagState*>addr
            free(s)
        free(self.parents)
        free(self.beam)


cdef TagState* extend_state(TagState* s, size_t clas, double* scores,
                            size_t nr_class):
    cdef double score, alt_score
    cdef size_t alt
    ext = <TagState*>calloc(1, sizeof(TagState))
    ext.prev = s
    ext.clas = clas
    ext.alt = 0
    if s == NULL:
        ext.score = 0
        ext.length = 0
    else:
        ext.score = s.score + scores[clas]
        ext.length = s.length + 1
        alt_score = 1
        for alt in range(nr_class):
            if alt == clas or alt == 0:
                continue
            score = scores[alt]
            if score > alt_score and alt != 0:
                ext.alt = alt
                alt_score = score
    return ext


cdef int fill_hist(size_t* hist, TagState* s, int t) except -1:
    while t >= 1 and s.prev != NULL:
        t -= 1
        hist[t] = s.clas
        s = s.prev

cdef size_t get_p(TagState* s):
    if s.prev == NULL:
        return 0
    else:
        return s.prev.clas


cdef size_t get_pp(TagState* s):
    if s.prev == NULL:
        return 0
    elif s.prev.prev == NULL:
        return 0
    else:
        return s.prev.prev.clas

