# cython: profile=True
"""
Handle parser features
"""
from libc.stdlib cimport malloc, free, calloc
from libc.stdint cimport uint64_t
from libcpp.pair cimport pair
import index.hashes
from cython.operator cimport dereference as deref, preincrement as inc
from _state cimport Kernel, Subtree

from io_parse cimport Sentence
#from index.hashes cimport encode_feat

from libcpp.vector cimport vector

from itertools import combinations

# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum

cdef enum:
    N0w
    N0p
    N0c
    N0cp
    N0lw
    N0lp
    N0lc
    N0lcp
    N0ll
    N0lv
    N0l2w
    N0l2p
    N0l2c
    N0l2cp
    N0l2l
    N1w
    N1p
    N1c
    N1cp
    N2w
    N2p
    N2c
    N2cp
    S0w
    S0p
    S0c
    S0cp
    S0l
    S0hw
    S0hp
    S0hc
    S0hcp
    S0hl
    S0hb
    S0lw
    S0lp
    S0lc
    S0lcp
    S0ll
    S0rw
    S0rp
    S0rc
    S0rcp
    S0rl
    S0l2w
    S0l2p
    S0l2c
    S0l2cp
    S0l2l
    S0l2b
    S0r2w
    S0r2p
    S0r2c
    S0r2cp
    S0r2l
    S0r2b
    S0h2w
    S0h2p
    S0h2c
    S0h2cp
    S0h2l
    S0lv
    S0rv
    dist
    S0llabs
    S0rlabs
    N0llabs
    N0orth
    N0paren
    N0quote
    N1orth
    N1paren
    N1quote
    S0re_orth
    S0re_w
    S0re_p
    S0re_c
    S0re_cp
    N0le_orth
    N0le_w
    N0le_p
    N0le_c
    N0le_cp
    CONTEXT_SIZE


def get_kernel_tokens():
    return [S0w, N0w, N1w, N2w, N0lw, N0l2w, S0hw, S0h2w, S0rw, S0r2w, S0lw, S0l2w]

def unigram(word, add_clusters=True):
    w = 1000
    pos = word + 1
    cluster = word + 2
    cluster_prefix = word + 3
    basic = ((w, word, pos), (w, word), (w, pos))
    clusters = ((w, word, pos, cluster_prefix), (w, word, pos, cluster),
                (w, word, cluster), (w, word, cluster_prefix),
                (w, pos, cluster), (w, pos, cluster_prefix))
    if add_clusters:
        return basic + clusters
    else:
        return basic


def _bigram(a, b, add_clusters=True):
    w1 = a
    p1 = a + 1
    c1 = a + 2
    cp1 = a + 3
    w2 = b
    p2 = b + 1
    c2 = b + 2
    cp2 = b + 3
    basic = ((w1, w2), (w1, p1, w2, p2), (p1, p2), (w1, p2), (p1, w2))
    clusters = ((c1, c2), (c1, p1, c2, p2), (c1, w1, cp2, p2),
                (cp1, p1, c2, p2), (cp1, p1, cp2, p2))
    if add_clusters:
        return basic + clusters
    else:
        return basic

def bigram_no_clusters(a, b):
    return _bigram(a, b, False)

def bigram_with_clusters(a, b):
    return _bigram(a, b, True)

def _trigram(a, b, c, add_clusters=True):
    w1 = a
    p1 = a + 1
    c1 = a + 2
    cp1 = a + 3
    w2 = b
    p2 = b + 1
    c2 = b + 2
    cp2 = b + 3
    w3 = c
    p3 = c + 1
    c3 = c + 2
    cp3 = c + 3

    basic = ((w1, w2, w3), (w1, w2, p3), (w1, p2, w3), (p1, w2, w3),
            (w1, p2, p3), (p1, w2, p3), (p1, p2, w3), (p1, p2, p3))
    clusters = ((c1, c2, p3), (c1, p2, w3), (p1, c2, c3), (c1, p2, p3),
                (p1, c2, p3), (p1, c2, c3), (p1, p2, p3))

    if add_clusters:
        return basic + clusters
    else:
        return basic


def trigram_no_clusters(a, b, c):
    return _trigram(a, b, c, False)

def trigram_with_clusters(a, b, c):
    return _trigram(a, b, c, True)


cdef void fill_context(size_t* context, size_t nr_label, size_t* words, size_t* pos,
                       size_t* clusters, size_t* cprefixes,
                       size_t* orths, size_t* parens, size_t* quotes,
                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l):
    context[N0w] = words[k.i]
    context[N0p] = pos[k.i]
    context[N0c] = clusters[k.i]
    context[N0cp] = cprefixes[k.i]

    context[N1w] = words[k.i + 1]
    context[N1p] = pos[k.i + 1]
    context[N1c] = clusters[k.i + 1]
    context[N1cp] = cprefixes[k.i + 1]

    context[N2w] = words[k.i + 2]
    context[N2p] = pos[k.i + 2]
    context[N2c] = clusters[k.i + 2]
    context[N2cp] = cprefixes[k.i + 2]

    context[S0w] = words[k.s0]
    context[S0p] = pos[k.s0]
    context[S0c] = clusters[k.s0]
    context[S0cp] = cprefixes[k.s0]
    context[S0l] = k.Ls0

    context[S0hw] = words[k.hs0]
    context[S0hp] = pos[k.hs0]
    context[S0hc] = clusters[k.hs0]
    context[S0hcp] = cprefixes[k.hs0]
    context[S0hl] = k.Lhs0
    context[S0hb] = k.hs0 != 0

    context[S0h2w] = words[k.h2s0]
    context[S0h2p] = pos[k.h2s0]
    context[S0h2c] = clusters[k.h2s0]
    context[S0h2cp] = cprefixes[k.h2s0]
    context[S0h2l] = k.Lh2s0
 
    context[S0lv] = s0l.val
    context[S0rv] = s0r.val
    context[N0lv] = n0l.val

    context[S0lw] = words[s0l.idx[0]]
    context[S0lp] = pos[s0l.idx[0]]
    context[S0lc] = clusters[s0l.idx[0]]
    context[S0lcp] = cprefixes[s0l.idx[0]]

    context[S0rw] = words[s0r.idx[0]]
    context[S0rp] = pos[s0r.idx[0]]
    context[S0rc] = clusters[s0r.idx[0]]
    context[S0rcp] = cprefixes[s0r.idx[0]]

    context[S0l2w] = words[s0l.idx[1]]
    context[S0l2p] = pos[s0l.idx[1]]
    context[S0l2c] = clusters[s0l.idx[1]]
    context[S0l2cp] = cprefixes[s0l.idx[1]]

    context[S0r2w] = words[s0r.idx[1]]
    context[S0r2p] = pos[s0r.idx[1]]
    context[S0r2c] = clusters[s0r.idx[1]]
    context[S0r2cp] = cprefixes[s0r.idx[1]]

    context[N0lw] = words[n0l.idx[0]]
    context[N0lp] = pos[n0l.idx[0]]
    context[N0lc] = clusters[n0l.idx[0]]
    context[N0lcp] = cprefixes[n0l.idx[0]]

    context[N0l2w] = words[n0l.idx[1]]
    context[N0l2p] = pos[n0l.idx[1]]
    context[N0l2c] = clusters[n0l.idx[1]]
    context[N0l2cp] = cprefixes[n0l.idx[1]]

    context[S0ll] = s0l.lab[0]
    context[S0l2l] = s0l.lab[1]
    context[S0rl] = s0r.lab[0]
    context[S0r2l] = s0r.lab[1]
    context[N0ll] = n0l.lab[0]
    context[N0l2l] = n0l.lab[1]

    context[S0llabs] = 0
    context[S0rlabs] = 0
    context[N0llabs] = 0
    cdef size_t i
    for i in range(4):
        context[S0llabs] += s0l.lab[i] << (nr_label - s0l.lab[i])
        context[S0rlabs] += s0r.lab[i] << (nr_label - s0r.lab[i])
        context[N0llabs] += n0l.lab[i] << (nr_label - n0l.lab[i])
    # TODO: Seems hard to believe we want to keep d non-zero when there's no
    # stack top. Experiment with this futrther.
    if k.s0 != 0:
        context[dist] = k.i - k.s0
    else:
        context[dist] = 0
    context[N0orth] = orths[k.i]
    context[N1orth] = orths[k.i + 1]
    context[N0paren] = parens[k.i]
    context[N1paren] = parens[k.i + 1]
    context[N0quote] = quotes[k.i]
    context[N1quote] = quotes[k.i + 1]
    context[N0le_orth] = orths[k.n0ledge]
    context[N0le_w] = words[k.n0ledge]
    context[N0le_p] = pos[k.n0ledge]
    context[N0le_c] = clusters[k.n0ledge]
    context[N0le_cp] = cprefixes[k.n0ledge]
    # TODO: This isn't accurate!!
    context[S0re_orth] = orths[k.n0ledge - 1]
    context[S0re_w] = words[k.n0ledge - 1]
    context[S0re_p] = pos[k.n0ledge - 1]
    context[S0re_c] = clusters[k.n0ledge - 1]
    context[S0re_cp] = cprefixes[k.n0ledge - 1]
 

cdef class FeatureSet:
    def __cinit__(self, nr_label, bint add_extra=False, ngrams=None, add_clusters=False):
        if ngrams is None:
            ngrams = []
        self.ngrams = ngrams
        self.add_clusters = add_clusters
        self.ngrams.append(-1)
        self.nr_label = nr_label
        # Sets "n"
        self._make_predicates(add_extra, ngrams, add_clusters)
        # TODO: Reference index.hashes constant
        self.nr_tags = 100 if add_extra else 0
        self.context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        self.features = <uint64_t*>calloc(self.n + self.nr_tags, sizeof(uint64_t))

    def __dealloc__(self):
        free(self.context)
        free(self.features)
        free(self.predicates)

    cdef uint64_t* extract(self, Sentence* sent, Kernel* k) except NULL:
        cdef size_t* context = self.context
        assert <size_t>k != 0
        fill_context(context, self.nr_label, sent.words, sent.pos,
                     sent.clusters, sent.cprefixes,
                     sent.orths, sent.parens, sent.quotes,
                     k, &k.s0l, &k.s0r, &k.n0l)
        cdef size_t i, j
        cdef uint64_t hashed
        cdef uint64_t value
        cdef uint64_t* features = self.features
        cdef bint seen_non_zero
        cdef Predicate* pred
        cdef size_t f = 0
        for i in range(self.n):
            pred = &self.predicates[i]
            seen_non_zero = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                pred.raws[j] = value
                if value != 0:
                    seen_non_zero = True
            if seen_non_zero:
                pred.raws[pred.n] = pred.id
                hashed = MurmurHash64A(pred.raws, (pred.n + 1) * sizeof(uint64_t), i)
                features[f] = hashed
                f += 1
        cdef size_t tag, letter
        #cdef bint* seen_tags 
        #if self.nr_tags and k.s0 > 0 and (k.s0 + 1) < k.i:
        #    seen_tags = <bint*>calloc(self.nr_tags, sizeof(bint))
        #    for i in range(k.s0 + 1, k.i):
        #        tag = sent.pos[i]
        #        if not seen_tags[tag]:
        #            features[f] = sent.pos[i]
        #            seen_tags[tag] = 1
        #            f += 1
        #    free(seen_tags)
        features[f] = 0
        return features

    def _make_predicates(self, bint add_extra, object to_add, add_clusters):
        feats = self._get_feats(add_extra, to_add, add_clusters)
        self.n = len(feats)
        self.predicates = <Predicate*>malloc(self.n * sizeof(Predicate))
        cdef Predicate pred
        for id_, args in enumerate(feats):
            size = args[0]
            args = args[1:]
            pred = Predicate(id=id_, n=len(args), expected_size = size)
            pred.raws = <uint64_t*>malloc((len(args) + 1) * sizeof(uint64_t))
            pred.args = <int*>malloc(len(args) * sizeof(int))
            for i, element in enumerate(sorted(args)):
                pred.args[i] = element
            self.predicates[id_] = pred

    def _get_feats(self, bint add_extra, object to_add, bint add_clusters):

        from_single = (
            (S0w, S0p),
            (S0w,),
            (S0p,),
            (N0w, N0p),
            (N0w,),
            (N0p,),
            (N1w, N1p),
            (N1w,),
            (N1p,),
            (N2w, N2p),
            (N2w,),
            (N2p,)
        )

        from_word_pairs = (
            (S0w, S0p, N0w, N0p),
            (S0w, S0p, N0w),
            (S0w, N0w, N0p),
            (S0w, S0p, N0p),
            (S0p, N0w, N0p),
            (S0w, N0w),
            (S0p, N0p),
            (N0p, N1p)
        )

        from_three_words = (
            (N0p, N1p, N2p),
            (S0p, N0p, N1p),
            (S0hp, S0p, N0p),
            (S0p, S0lp, N0p),
            (S0p, S0rp, N0p),
            (S0p, N0p, N0lp)
        )

        distance = (
            (dist, S0w),
            (dist, S0p),
            (dist, N0w),
            (dist, N0p),
            (dist, S0w, N0w),
            (dist, S0p, N0p),
        )

        valency = (
            (S0w, S0rv),
            (S0p, S0rv),
            (S0w, S0lv),
            (S0p, S0lv),
            (N0w, N0lv),
            (N0p, N0lv),
        )

        zhang_unigrams = (
            (S0hw,),
            (S0hp,),
            (S0lw,),
            (S0lp,),
            (S0rw,),
            (S0rp,),
            (N0lw,),
            (N0lp,),
        )

        third_order = (
            (S0h2w,),
            (S0h2p,),
            (S0l2w,),
            (S0l2p,),
            (S0r2w,),
            (S0r2p,),
            (N0l2w,),
            (N0l2p,),
            (S0p, S0lp, S0l2p),
            (S0p, S0rp, S0r2p),
            (S0p, S0hp, S0h2p),
            (N0p, N0lp, N0l2p)
        )

        labels = (
            (S0l,),
            (S0ll,),
            (S0rl,),
            (N0ll,),
            (S0hl,),
            (S0l2l,),
            (S0r2l,),
            (N0l2l,),
        )
        label_sets = (
            (S0w, S0rlabs),
            (S0p, S0rlabs),
            (S0w, S0llabs),
            (S0p, S0llabs),
            (N0w, N0llabs),
            (N0p, N0llabs),
        )

        extra = (
            (S0p, N0p, S0ll),
            (S0w, N0p, S0ll),
            (S0p, N0p, S0rp, S0ll),
            (S0w, S0rw),
            (S0rw, N0p),
            (S0rw, N0w),
            (S0p, S0rw, N0p),
            (S0w, S0rw, N0w),
            (N0orth),
            (N1orth),
            (N0paren),
            (N1paren),
            (N0quote),
            (N1quote),
            (S0rw, N0orth),
            (S0rw, N0orth, N1orth),
            (S0re_p, N0le_orth),
            (S0re_p, N0le_p),
            (S0re_p, N0le_w),
            (S0re_p, N0le_p, N0p)
        )

        unigrams = (
            unigram(S0w)
            + unigram(S0hw)
            + unigram(S0h2w)
            + unigram(S0rw)
            + unigram(S0r2w)
            + unigram(S0lw)
            + unigram(S0l2w)
            + unigram(N0w)
            + unigram(N1w)
            + unigram(N2w)
            + unigram(N0lw)
            + unigram(N0l2w)
        )

        if add_extra:
            print "Add extra feats"
            feats = unigrams + distance + valency + labels + label_sets
            kernel_tokens = get_kernel_tokens()
            
            bigram = bigram_with_clusters if add_clusters else bigram_no_clusters
            trigram = trigram_with_clusters if add_clusters else trigram_no_clusters

            for i, (t1, t2) in enumerate(combinations(kernel_tokens, 2)):
                if i in to_add:
                    feats += bigram(t1, t2)
            for i, (t1, t2, t3) in enumerate(combinations(kernel_tokens, 3)):
                if i in to_add:
                    feats += trigram(t1, t2, t3)
        else:
            feats = from_single + from_word_pairs + from_three_words + distance
            feats += valency + zhang_unigrams + third_order
            feats += labels
            feats += label_sets

        #assert len(set(feats)) == len(feats), '%d vs %d' % (len(set(feats)), len(feats))
        return feats


