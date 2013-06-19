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

from libcpp.vector cimport vector

from itertools import combinations

# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum
# Ensure each token's attributes are listed: w, p, c, cp
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
    
    N3w
    N3p
    N3c
    N3cp
    
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

    S0l0w
    S0l0p
    S0l0c
    S0l0cp

    S0r0w
    S0r0p
    S0r0c
    S0r0cp

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
    return [S0w, N0w, N1w, N2w, N0lw, N0l2w, S0hw, S0h2w, S0rw, S0r2w, S0lw,
            S0l2w, S0re_w, N0le_w, N3w, S0l0w, S0r0w]

def get_best_bigrams(all_bigrams, n=50):
    best = [0, 26, 12, 126, 1, 5, 41, 16, 40, 86, 20, 87, 18, 27, 22, 30,
            3, 104, 24, 65, 117, 132, 29, 11, 34, 131, 7, 116, 32, 36, 81,
            15, 9, 21, 44, 6, 128, 95, 89, 17, 96, 38, 19, 84, 14, 43, 4,
            2, 82, 90, 54, 76, 58, 77, 53, 23, 13, 31, 28, 42, 101, 35, 111,
            121, 122, 25, 10, 127, 106, 129, 130, 33, 120, 37, 100, 66, 135,
            59, 110, 8, 61, 107][:n]
    return [all_bigrams[i] for i in best]

def get_best_trigrams(all_trigrams, n=25):
    best = [2, 199, 158, 61, 66, 5, 150, 1, 88, 154, 85, 25, 53, 10, 3, 60,
            73, 175, 114, 4, 6, 148, 205, 197, 0, 71, 127, 200, 142, 84, 43,
            89, 45, 95, 33, 110, 182, 20, 24, 159, 51, 106, 26, 8, 178, 151, 12,
            166, 192, 7, 190, 147, 13, 194, 50, 129, 174][:25]
    return [all_trigrams[i] for i in best][:n]


def unigram(word, add_clusters=False):
    pos = word + 1
    cluster = word + 2
    cluster_prefix = word + 3
    basic = ((word, pos), (word,), (pos,))
    clusters = ((word, pos, cluster_prefix), (word, pos, cluster),
                (word, cluster), (word, cluster_prefix),
                (pos, cluster), (pos, cluster_prefix))
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
    basic = ((w1, w2), (p1, p2), (w1, p2), (p1, w2))
    clusters = ((c1, c2), (cp1, p1, cp2, p2))
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

    basic = ((w1, w2, w3), (w1, p2, p3), (p1, w2, p3), (p1, p2, w3), (p1, p2, p3))
    clusters = ((c1, c2, c3), (cp1, p1, cp2, p2, cp3, p3))
    #clusters = ((c1, c2, p3), (c1, p2, w3), (p1, c2, c3), (c1, p2, p3),
    #            (p1, c2, p3), (p1, c2, c3), (p1, p2, p3))

    if add_clusters:
        return basic + clusters
    else:
        return basic


def trigram_no_clusters(a, b, c):
    return _trigram(a, b, c, False)

def trigram_with_clusters(a, b, c):
    return _trigram(a, b, c, True)


cdef void fill_context(size_t* context, size_t nr_label, size_t* words,
                       size_t* clusters, size_t* cprefixes,
                       size_t* orths, size_t* parens, size_t* quotes,
                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l):
    context[N0w] = words[k.i]
    context[N0p] = k.n0p
    context[N0c] = clusters[k.i]
    context[N0cp] = cprefixes[k.i]

    context[N1w] = words[k.i + 1]
    context[N1p] = k.n1p
    context[N1c] = clusters[k.i + 1]
    context[N1cp] = cprefixes[k.i + 1]

    context[N2w] = words[k.i + 2]
    context[N2p] = k.n2p
    context[N2c] = clusters[k.i + 2]
    context[N2cp] = cprefixes[k.i + 2]

    context[N3w] = words[k.i + 3]
    context[N3p] = k.n3p
    context[N3c] = clusters[k.i + 3]
    context[N3cp] = cprefixes[k.i + 3]

    context[S0w] = words[k.s0]
    context[S0p] = k.s0p
    context[S0c] = clusters[k.s0]
    context[S0cp] = cprefixes[k.s0]
    context[S0l] = k.Ls0

    context[S0hw] = words[k.hs0]
    context[S0hp] = k.hs0p
    context[S0hc] = clusters[k.hs0]
    context[S0hcp] = cprefixes[k.hs0]
    context[S0hl] = k.Lhs0
    context[S0hb] = k.hs0 != 0

    context[S0h2w] = words[k.h2s0]
    context[S0h2p] = k.h2s0p
    context[S0h2c] = clusters[k.h2s0]
    context[S0h2cp] = cprefixes[k.h2s0]
    context[S0h2l] = k.Lh2s0
 
    context[S0lv] = s0l.val
    context[S0rv] = s0r.val
    context[N0lv] = n0l.val

    context[S0lw] = words[s0l.idx[0]]
    context[S0lp] = s0l.tags[0]
    context[S0lc] = clusters[s0l.idx[0]]
    context[S0lcp] = cprefixes[s0l.idx[0]]

    context[S0rw] = words[s0r.idx[0]]
    context[S0rp] = s0r.tags[0]
    context[S0rc] = clusters[s0r.idx[0]]
    context[S0rcp] = cprefixes[s0r.idx[0]]

    context[S0l2w] = words[s0l.idx[1]]
    context[S0l2p] = s0l.tags[1]
    context[S0l2c] = clusters[s0l.idx[1]]
    context[S0l2cp] = cprefixes[s0l.idx[1]]

    context[S0r2w] = words[s0r.idx[1]]
    context[S0r2p] = s0r.tags[1]
    context[S0r2c] = clusters[s0r.idx[1]]
    context[S0r2cp] = cprefixes[s0r.idx[1]]

    context[S0l0w] = words[s0l.idx[2]]
    context[S0l0p] = s0l.tags[2]
    context[S0l0c] = clusters[s0l.idx[2]]
    context[S0l0cp] = cprefixes[s0l.idx[2]]

    context[S0r0w] = words[s0r.idx[2]]
    context[S0r0p] = s0r.tags[2]
    context[S0r0c] = clusters[s0r.idx[2]]
    context[S0r0cp] = cprefixes[s0r.idx[2]]

    context[N0lw] = words[n0l.idx[0]]
    context[N0lp] = n0l.tags[0]
    context[N0lc] = clusters[n0l.idx[0]]
    context[N0lcp] = cprefixes[n0l.idx[0]]

    context[N0l2w] = words[n0l.idx[1]]
    context[N0l2p] = n0l.tags[1]
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
    context[N0le_p] = k.n0ledgep
    context[N0le_c] = clusters[k.n0ledge]
    context[N0le_cp] = cprefixes[k.n0ledge]
    context[S0re_orth] = orths[k.s0redge]
    context[S0re_w] = words[k.s0redge]
    context[S0re_p] = k.s0redgep
    context[S0re_c] = clusters[k.s0redge]
    context[S0re_cp] = cprefixes[k.s0redge]
 

cdef class FeatureSet:
    def __cinit__(self, nr_label, uint64_t mask_value=1, feat_set="zhang",
                  ngrams=None, add_clusters=False):
        if ngrams is None:
            ngrams = []
        self.name = feat_set
        self.ngrams = ngrams
        self.add_clusters = add_clusters
        if add_clusters:
            print "Adding cluster feats"
        self.nr_label = nr_label
        # Value that indicates the value has been "masked", e.g. it was pruned
        # as a rare word. If a feature contains any masked values, it is dropped.
        self.mask_value = mask_value
        # Sets "n"
        self._make_predicates(self.name, ngrams, add_clusters)
        self.context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        self.features = <uint64_t*>calloc(self.n, sizeof(uint64_t))

    def __dealloc__(self):
        free(self.context)
        free(self.features)
        free(self.predicates)

    cdef uint64_t* extract(self, Sentence* sent, Kernel* k) except NULL:
        cdef:
            size_t i, j, f, size
            uint64_t value
            bint seen_non_zero, seen_masked
            Predicate* pred

        cdef size_t* context = self.context
        cdef uint64_t* features = self.features
        fill_context(context, self.nr_label, sent.words,
                     sent.clusters, sent.cprefixes,
                     sent.orths, sent.parens, sent.quotes,
                     k, &k.s0l, &k.s0r, &k.n0l)
        f = 0
        for i in range(self.n):
            pred = &self.predicates[i]
            seen_non_zero = False
            seen_masked = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                if value == self.mask_value:
                    seen_masked = True
                    break
                elif value != 0:
                    seen_non_zero = True
                pred.raws[j] = value
            if seen_non_zero and not seen_masked:
                pred.raws[pred.n] = pred.id
                size = (pred.n + 1) * sizeof(uint64_t)
                features[f] = MurmurHash64A(pred.raws, size, i)
                f += 1
        features[f] = 0
        return features

    def _make_predicates(self, object name, object ngrams, add_clusters):
        feats = self._get_feats(name, ngrams, add_clusters)
        self.n = len(feats)
        self.predicates = <Predicate*>malloc(self.n * sizeof(Predicate))
        cdef Predicate pred
        for id_, args in enumerate(feats):
            pred = Predicate(id=id_, n=len(args), expected_size = 1000)
            pred.raws = <uint64_t*>malloc((len(args) + 1) * sizeof(uint64_t))
            pred.args = <int*>malloc(len(args) * sizeof(int))
            for i, element in enumerate(sorted(args)):
                pred.args[i] = element
            self.predicates[id_] = pred

    def _get_feats(self, object feat_level, object ngrams, bint add_clusters):

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
            unigram(S0w, add_clusters)
            + unigram(S0hw, add_clusters)
            + unigram(S0h2w, add_clusters)
            + unigram(S0rw, add_clusters)
            + unigram(S0r2w, add_clusters)
            + unigram(S0lw, add_clusters)
            + unigram(S0l2w, add_clusters)
            + unigram(N0w, add_clusters)
            + unigram(N1w, add_clusters)
            + unigram(N2w, add_clusters)
            + unigram(N0lw, add_clusters)
            + unigram(N0l2w, add_clusters)
            + unigram(S0re_w, add_clusters)
            + unigram(N0le_w, add_clusters)
            + unigram(N3w, add_clusters)
            + unigram(S0l0w, add_clusters)
            + unigram(S0r0w, add_clusters)
        )
        if feat_level == 'zhang':
            print "Use Zhang feats"
            feats = from_single + from_word_pairs + from_three_words + distance
            feats += valency + zhang_unigrams + third_order
            feats += labels
            feats += label_sets
        else:
            print "Use %d ngram feats" % len(ngrams)
            feats = tuple(unigrams)
            kernel_tokens = get_kernel_tokens()
            bigram = bigram_with_clusters if add_clusters else bigram_no_clusters
            trigram = trigram_with_clusters if add_clusters else trigram_no_clusters
            for ngram_feat in ngrams:
                if len(ngram_feat) == 2:
                    feats += bigram(*ngram_feat)
                elif len(ngram_feat) == 3:
                    feats += trigram(*ngram_feat)
                else:
                    raise StandardError, ngram_feat
            if feat_level != 'iso':
                feats += valency
                feats += distance
                feats += label_sets
                feats += labels
            else:
                print "No extra feats"
            if feat_level == 'full':
                print "Full feats"
                feats += from_single
                feats += zhang_unigrams
                feats += third_order
                feats += from_word_pairs
                feats += from_three_words
        # Sort each feature, and sort and unique the set of them
        return tuple(sorted(set([tuple(sorted(f)) for f in feats])))
