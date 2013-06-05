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

import itertools

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
    ww = 100000
    pp = 2500
    pw = 50000
    ppp = 10000
    
    w1 = a
    p1 = a + 1
    c1 = a + 2
    cp1 = a + 3
    w2 = b
    p2 = b + 1
    c2 = b + 2
    cp2 = b + 3
    basic = ((ww, w1, w2), (ww, w1, p1, w2, p2), (ww, p1, p2), (ww, w1, p2), (ww, p1, w2))
    clusters = ((pp, c1, c2), (pp, c1, p1, c2, p2), (ww, c1, w1, cp2, p2),
                (ww, cp1, p1, c2, p2), (ww, cp1, p1, cp2, p2))
    if add_clusters:
        return basic + clusters
    else:
        return basic

def bigram_no_clusters(a, b):
    return _bigram(a, b, False)

def bigram_with_clusters(a, b):
    return _bigram(a, b, True)

def _trigram(a, b, c, add_clusters=True):
    ww = 100000
    pp = 2500
    pw = 50000
    ppp = 10000
    
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

    basic = ((ww, w1, w2, w3), (ww, w1, w2, p3), (ww, w1, p2, w3), (ww, p1, w2, w3),
            (ww, w1, p2, p3), (ww, p1, w2, p3), (ww, p1, p2, w3), (ww, p1, p2, p3))
    clusters = ((ww, c1, c2, p3), (ww, c1, p2, w3), (ww, p1, c2, c3), (ww, c1, p2, p3),
             (ww, p1, c2, p3), (ww, p1, c2, c3), (ww, p1, p2, p3))

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
    def __cinit__(self, nr_label, bint add_extra=False, to_add=None, add_clusters=False):
        if to_add is None:
            to_add = range(66)
        self.bigrams = to_add
        self.bigrams.append(-1)
        self.nr_label = nr_label
        # Sets "n"
        self._make_predicates(add_extra, to_add, add_clusters)
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
        cdef bint* seen_tags 
        if self.nr_tags and k.s0 > 0 and (k.s0 + 1) < k.i:
            seen_tags = <bint*>calloc(self.nr_tags, sizeof(bint))
            for i in range(k.s0 + 1, k.i):
                tag = sent.pos[i]
                if not seen_tags[tag]:
                    features[f] = sent.pos[i]
                    seen_tags[tag] = 1
                    f += 1
            free(seen_tags)
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
        # For multi-part features, we want expected sizes, as the table will
        # resize
        wp = 50000
        wwpp = 100000
        wwp = 100000
        wpp = 100000
        ww = 100000
        pp = 2500
        pw = 50000
        ppp = 10000
        vw = 60000
        vp = 200
        lwp = 110000
        lww = 110000
        lw = 80000
        lp = 500
        dp = 500
        dw = 50000
        dppp = 10000
        dpp = 30000
        dww = 60000
        # For unigrams we need max values
        w = 20000
        p = 100
        l = 100

        from_single = (
            (wp, S0w, S0p),
            (w, S0w,),
            (p, S0p,),
            (wp, N0w, N0p),
            (w, N0w,),
            (p, N0p,),
            (wp, N1w, N1p),
            (w, N1w,),
            (p, N1p,),
            (wp, N2w, N2p),
            (w, N2w,),
            (p, N2p,)
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
        bigram = bigram_with_clusters if add_clusters else bigram_no_clusters
        trigram = trigram_with_clusters if add_clusters else trigram_no_clusters
        s0_bigrams = (
            bigram(S0w, N0w)
            + bigram(S0w, N1w)
            + bigram(S0w, N2w)
            + bigram(S0w, N0lw)
            + bigram(S0w, N0l2w)
        )
        
        s0h_bigrams = (
            bigram(S0hw, N0w)
            + bigram(S0hw, N1w)
            + bigram(S0hw, N2w)
            + bigram(S0h2w, N0w)
        )
        s0r_bigrams = (
            bigram(S0rw, N0w)
            + bigram(S0r2w, N0w)
            + bigram(S0rw, S0r2w)
        )
        s0l_bigrams = (
            bigram(S0lw, N0w)
            + bigram(S0l2w, N0w)
            + bigram(S0lw, S0l2w)
        )
        n_bigrams = (
            bigram(N0w, N1w)
            + bigram(N1w, N2w)
            + bigram(N0w, N0lw)
            + bigram(N0lw, N0l2w)
        )

        trigrams = (
            trigram(S0hw, S0w, N0w)
            + trigram(S0h2w, S0w, N0w)
            + trigram(S0h2w, S0h2w, S0w)
            + trigram(N0w, N1w, N2w)
            + trigram(S0w, N0w, N1w)
            + trigram(S0w, N0w, N0lw)
            + trigram(N0w, N0lw, N0l2w)
            + trigram(S0w, S0rw, N0w)
            + trigram(S0w, S0r2w, S0rw)
            + trigram(S0w, S0lw, S0l2w)
            + trigram(S0hw, S0w, S0rw)
            + trigram(S0hw, S0w, S0lw)
            + trigram(S0hw, S0w, N0lw)
        )

        from_word_pairs = (
            (wwpp, S0w, S0p, N0w, N0p),
            (wwp, S0w, S0p, N0w),
            (wwp, S0w, N0w, N0p),
            (wpp, S0w, S0p, N0p),
            (wpp, S0p, N0w, N0p),
            (ww, S0w, N0w),
            (pp, S0p, N0p),
            (pp, N0p, N1p)
        )

        from_three_words = (
            (ppp, N0p, N1p, N2p),
            (ppp, S0p, N0p, N1p),
            (ppp, S0hp, S0p, N0p),
            (ppp, S0p, S0lp, N0p),
            (ppp, S0p, S0rp, N0p),
            (ppp, S0p, N0p, N0lp)
        )

        distance = (
            (dw, dist, S0w),
            (dp, dist, S0p),
            (dw, dist, N0w),
            (dp, dist, N0p),
            (dww, dist, S0w, N0w),
            (dpp, dist, S0p, N0p),
        )

        valency = (
            (vw, S0w, S0rv),
            (vp, S0p, S0rv),
            (vw, S0w, S0lv),
            (vp, S0p, S0lv),
            (vw, N0w, N0lv),
            (vp, N0p, N0lv),
        )

        zhang_unigrams = (
            (w, S0hw,),
            (p, S0hp,),
            (w, S0lw,),
            (p, S0lp,),
            (w, S0rw,),
            (p, S0rp,),
            (w, N0lw,),
            (p, N0lp,),
        )

        third_order = (
            (w, S0h2w,),
            (p, S0h2p,),
            (w, S0l2w,),
            (p, S0l2p,),
            (w, S0r2w,),
            (p, S0r2p,),
            (w, N0l2w,),
            (p, N0l2p,),
            (ppp, S0p, S0lp, S0l2p),
            (ppp, S0p, S0rp, S0r2p),
            (ppp, S0p, S0hp, S0h2p),
            (ppp, N0p, N0lp, N0l2p)
        )

        labels = (
            (l, S0l,),
            (l, S0ll,),
            (l, S0rl,),
            (l, N0ll,),
            (l, S0hl,),
            (l, S0l2l,),
            (l, S0r2l,),
            (l, N0l2l,),
        )
        label_sets = (
            (lw, S0w, S0rlabs),
            (lp, S0p, S0rlabs),
            (lw, S0w, S0llabs),
            (lp, S0p, S0llabs),
            (lw, N0w, N0llabs),
            (lp, N0p, N0llabs),
        )

        extra = (
            (ppp, S0p, N0p, S0ll),
            (wpp, S0w, N0p, S0ll),
            (wwp, S0p, N0p, S0rp, S0ll),
            (ww, S0w, S0rw),
            (wp, S0rw, N0p),
            (ww, S0rw, N0w),
            (wpp, S0p, S0rw, N0p),
            (wwp, S0w, S0rw, N0w),
            (p, N0orth),
            (p, N1orth),
            (p, N0paren),
            (p, N1paren),
            (p, N0quote),
            (p, N1quote),
            (pp, S0rw, N0orth),
            (ppp, S0rw, N0orth, N1orth),
            (wp, S0re_p, N0le_orth),
            (pp, S0re_p, N0le_p),
            (wp, S0re_p, N0le_w),
            (ppp, S0re_p, N0le_p, N0p)
        )
        if add_extra:
            print "Add extra feats"
            feats = unigrams + distance + valency + labels + label_sets
            pairs = itertools.combinations((S0w, N0w, N1w, N2w, N0lw, N0l2w, S0hw,
                                            S0h2w, S0rw, S0r2w, S0lw, S0l2w), 2)

            for i, (t1, t2) in enumerate(pairs):
                if i in to_add:
                    feats += bigram(t1, t2)
            feats += from_three_words
            #if 's0' in to_add:
            #    feats += s0_bigrams
            #if 's0h' in to_add:
            #    feats += s0h_bigrams
            #if 's0r' in to_add:
            #    feats += s0r_bigrams
            #if 's0l' in to_add:
            #    feats += s0l_bigrams
            #if 'n' in to_add:
            #    feats += n_bigrams
            #if 'tri' in to_add:
            #    feats += trigrams
            #feats += trigrams
        else:
            feats = from_single + from_word_pairs + from_three_words + distance + valency + zhang_unigrams + third_order
            feats += labels
            feats += label_sets

        #assert len(set(feats)) == len(feats), '%d vs %d' % (len(set(feats)), len(feats))
        return feats


