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
    N0c6
    N0c4

    N0lw
    N0lp
    N0lc
    N0lc6
    N0lc4
    
    N0ll
    N0lv
    
    N0l2w
    N0l2p
    N0l2c
    N0l2c6
    N0l2c4
    
    N0l2l
    
    N1w
    N1p
    N1c
    N1c6
    N1c4
    
    N2w
    N2p
    N2c
    N2c6
    N2c4
    
    N3w
    N3p
    N3c
    N3c6
    N3c4
    
    S0w
    S0p
    S0c
    S0c6
    S0c4
    
    S0l
    
    S0hw
    S0hp
    S0hc
    S0hc6
    S0hc4
    
    S0hl

    S0lw
    S0lp
    S0lc
    S0lc6
    S0lc4
    
    S0ll
    
    S0rw
    S0rp
    S0rc
    S0rc6
    S0rc4
    
    S0rl
    
    S0l2w
    S0l2p
    S0l2c
    S0l2c6
    S0l2c4
    
    S0l2l

    S0r2w
    S0r2p
    S0r2c
    S0r2c6
    S0r2c4
    
    S0r2l

    S0l0w
    S0l0p
    S0l0c
    S0l0c6
    S0l0c4

    S0r0w
    S0r0p
    S0r0c
    S0r0c6
    S0r0c4

    S0h2w
    S0h2p
    S0h2c
    S0h2c6
    S0h2c4
    
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
    S0re_c6
    S0re_c4
    
    N0le_orth
    
    N0le_w
    N0le_p
    N0le_c
    N0le_c6
    N0le_c4

    prev_edit
    prev_edit_wmatch
    prev_edit_pmatch
    prev_edit_word
    prev_edit_pos
    prev_prev_edit

    wcopy 
    pcopy

    wexact
    pexact

    CONTEXT_SIZE


#def get_kernel_tokens():
#    return [S0w, N0w, N1w, N2w, N0lw, N0l2w, S0hw, S0h2w, S0rw, S0r2w, S0lw,
#            S0l2w, S0re_w, N0le_w, N3w, S0l0w, S0r0w]

def get_kernel_tokens():
    return [S0w, N0w, N1w, N2w, N0lw, N0l2w, S0hw, S0h2w, S0rw, S0r2w, S0lw,
            S0l2w]

def get_best_bigrams(all_bigrams, n=0):
    return []

def get_best_trigrams(all_trigrams, n=0):
    return []
    #return [all_trigrams[i] for i in best][:n]

def get_best_features():
    # s0_n0 n0_s0re s0_n0le s0_n0_n0l s0_n1 s0_s0h n1_n0le n0_n0l_s0r0
    # s0_n0l2_n0le s0_s0r_s0l s0_n1_n0le s0_n0l2_s0re n0_s0h_s0re n0_n1_s0l
    # n0_s0h2 s0_n1_s0r s0_n0_n3
    return []

def unigram(word, add_clusters=False):
    assert word >= 0
    assert word < (CONTEXT_SIZE - 5)
 
    pos = word + 1
    cluster = word + 2
    cluster6 = word + 3
    cluster4 = word + 4
    basic = ((word, pos), (word,), (pos,))
    clusters = ((cluster,), (cluster6,), (cluster4,),
                (pos, cluster), (pos, cluster6), (pos, cluster4),
                (word, cluster6), (word, cluster4))
    if add_clusters:
        return basic + clusters
    else:
        return basic


def _bigram(a, b, add_clusters=True):
    assert a >= 0
    assert b >= 0
    assert a < (CONTEXT_SIZE - 5)
    assert b < (CONTEXT_SIZE - 5)
    w1 = a
    p1 = a + 1
    c1 = a + 2
    c6_1 = a + 3
    c4_1 = a + 4
    w2 = b
    p2 = b + 1
    c2 = b + 2
    c6_2 = b + 3
    c4_2 = b + 4
    basic = ((w1, w2), (p1, p2), (p1, w2), (w1, p2))
    clusters = ((c1, c2), (c1, w2), (w1, c2), (c6_1, c6_2), (c4_1, c4_2),
                (c6_1, p1, p2), (p1, c6_2, p2), (c4_1, p1, w2),
                (w1, c4_2, p2))
    if add_clusters:
        return basic + clusters
    else:
        return basic

def bigram_no_clusters(a, b):
    return _bigram(a, b, False)

def bigram_with_clusters(a, b):
    return _bigram(a, b, True)

def _trigram(a, b, c, add_clusters=True):
    assert a >= 0
    assert b >= 0
    assert c >= 0
    assert a < (CONTEXT_SIZE - 5)
    assert b < (CONTEXT_SIZE - 5)
    assert c < (CONTEXT_SIZE - 5)

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

    #basic = ((w1, w2, w3), (w1, p2, p3), (p1, w2, p3), (p1, p2, w3), (p1, p2, p3))
    basic = ((w1, w2, w3), (w1, p2, p3), (p1, w2, p3), (p1, p2, w3),
             (p1, p2, p3))
    #clusters = ((c1, c2, p3), (c1, p2, w3), (p1, c2, c3), (c1, p2, p3),
    #            (p1, c2, p3), (p1, c2, c3), (p1, p2, p3))
    clusters = ((c1, c2, c3), (c1, p2, p3), (cp1, p2, p3), (p1, c2, p3), (p1, cp2, p3),
                (p1, p2, c3), (p1, p2, cp3))

    if add_clusters:
        return basic + clusters
    else:
        return basic


def trigram_no_clusters(a, b, c):
    return _trigram(a, b, c, False)

def trigram_with_clusters(a, b, c):
    return _trigram(a, b, c, True)


cdef void fill_context(size_t* context, size_t nr_label, size_t* words,
                       size_t* tags,
                       size_t* clusters, size_t* cprefix6s, size_t* cprefix4s,
                       size_t* orths, size_t* parens, size_t* quotes,
                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l):
    context[N0w] = words[k.i]
    context[N0p] = k.n0p
    context[N0c] = clusters[k.i]
    context[N0c6] = cprefix6s[k.i]
    context[N0c4] = cprefix4s[k.i]

    context[N1w] = words[k.i + 1]
    context[N1p] = k.n1p
    context[N1c] = clusters[k.i + 1]
    context[N1c6] = cprefix6s[k.i + 1]
    context[N1c4] = cprefix4s[k.i + 1]

    context[N2w] = words[k.i + 2]
    context[N2p] = tags[k.i + 2]
    context[N2c] = clusters[k.i + 2]
    context[N2c6] = cprefix6s[k.i + 2]
    context[N2c4] = cprefix4s[k.i + 2]

    context[N3w] = words[k.i + 3]
    context[N3p] = k.n3p
    context[N3c] = clusters[k.i + 3]
    context[N3c6] = cprefix6s[k.i + 3]
    context[N3c4] = cprefix4s[k.i + 3]

    context[S0w] = words[k.s0]
    context[S0p] = k.s0p
    context[S0c] = clusters[k.s0]
    context[S0c6] = cprefix6s[k.s0]
    context[S0c4] = cprefix4s[k.s0]
    context[S0l] = k.Ls0

    context[S0hw] = words[k.hs0]
    context[S0hp] = k.hs0p
    context[S0hc] = clusters[k.hs0]
    context[S0hc6] = cprefix6s[k.hs0]
    context[S0hc4] = cprefix4s[k.hs0]
    context[S0hl] = k.Lhs0

    context[S0h2w] = words[k.h2s0]
    context[S0h2p] = k.h2s0p
    context[S0h2c] = clusters[k.h2s0]
    context[S0h2c6] = cprefix6s[k.h2s0]
    context[S0h2c4] = cprefix4s[k.h2s0]
    context[S0h2l] = k.Lh2s0
 
    context[S0lv] = s0l.val
    context[S0rv] = s0r.val
    context[N0lv] = n0l.val

    context[S0lw] = words[s0l.idx[0]]
    context[S0lp] = s0l.tags[0]
    context[S0lc] = clusters[s0l.idx[0]]
    context[S0lc6] = cprefix6s[s0l.idx[0]]
    context[S0lc4] = cprefix4s[s0l.idx[0]]

    context[S0rw] = words[s0r.idx[0]]
    context[S0rp] = s0r.tags[0]
    context[S0rc] = clusters[s0r.idx[0]]
    context[S0rc6] = cprefix6s[s0r.idx[0]]
    context[S0rc4] = cprefix4s[s0r.idx[0]]

    context[S0l2w] = words[s0l.idx[1]]
    context[S0l2p] = s0l.tags[1]
    context[S0l2c] = clusters[s0l.idx[1]]
    context[S0l2c6] = cprefix6s[s0l.idx[1]]
    context[S0l2c4] = cprefix4s[s0l.idx[1]]

    context[S0r2w] = words[s0r.idx[1]]
    context[S0r2p] = s0r.tags[1]
    context[S0r2c] = clusters[s0r.idx[1]]
    context[S0r2c6] = cprefix6s[s0r.idx[1]]
    context[S0r2c4] = cprefix4s[s0r.idx[1]]

    context[S0l0w] = words[s0l.idx[2]]
    context[S0l0p] = s0l.tags[2]
    context[S0l0c] = clusters[s0l.idx[2]]
    context[S0l0c6] = cprefix6s[s0l.idx[2]]
    context[S0l0c4] = cprefix4s[s0l.idx[2]]

    context[S0r0w] = words[s0r.idx[2]]
    context[S0r0p] = s0r.tags[2]
    context[S0r0c] = clusters[s0r.idx[2]]
    context[S0r0c6] = cprefix6s[s0r.idx[2]]
    context[S0r0c4] = cprefix6s[s0r.idx[2]]

    context[N0lw] = words[n0l.idx[0]]
    context[N0lp] = n0l.tags[0]
    context[N0lc] = clusters[n0l.idx[0]]
    context[N0lc6] = cprefix6s[n0l.idx[0]]
    context[N0lc4] = cprefix6s[n0l.idx[0]]

    context[N0l2w] = words[n0l.idx[1]]
    context[N0l2p] = n0l.tags[1]
    context[N0l2c] = clusters[n0l.idx[1]]
    context[N0l2c6] = cprefix6s[n0l.idx[1]]
    context[N0l2c4] = cprefix4s[n0l.idx[1]]

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
        assert k.i > k.s0
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
    context[N0le_c6] = cprefix6s[k.n0ledge]
    context[N0le_c4] = cprefix4s[k.n0ledge]
    
    context[S0re_p] = k.s0redgep
    if k.n0ledge > 0:
        context[S0re_orth] = orths[k.n0ledge - 1]
        context[S0re_w] = words[k.n0ledge - 1]
        context[S0re_c] = clusters[k.n0ledge - 1]
        context[S0re_c6] = cprefix6s[k.n0ledge - 1]
        context[S0re_c4] = cprefix4s[k.n0ledge - 1]
    else:
        context[S0re_w] = 0
        context[S0re_c] = 0
        context[S0re_c6] = 0
        context[S0re_c4] = 0
        context[S0re_orth] = 0
    if k.prev_edit and k.i != 0:
        context[prev_edit] = 1
        context[prev_edit_wmatch] = 1 if words[k.i - 1] == words[k.i] else 0
        context[prev_edit_pmatch] = 1 if k.prev_tag == tags[k.i] else 0
        context[prev_prev_edit] = 1 if k.prev_prev_edit else 0
        context[prev_edit_word] = words[k.i - 1]
        context[prev_edit_pos] = k.prev_tag
    else:
        context[prev_edit] = 0
        context[prev_edit_wmatch] = 0
        context[prev_edit_pmatch] = 0
        context[prev_prev_edit] = 0
        context[prev_edit_word] = 0
        context[prev_edit_pos] = 0
    # These features find how much of S0's span matches N0's span, starting from
    # the left. A 3-match span will fire features for 1-match, 2-match and 3-match.
    context[wcopy] = 0
    context[wexact] = 1
    for i in range(5):
        if ((k.n0ledge + i) > k.i) or ((k.s0ledge + i) >= k.n0ledge):
            break
        if words[k.n0ledge + i] == words[k.s0ledge + i]:
            context[wcopy] += 1
        else:
            context[wexact] = 0
            break
    context[pcopy] = 0
    context[pexact] = 1
    for i in range(5):
        if ((k.n0ledge + i) > k.i) or ((k.s0ledge + i) >= k.n0ledge):
            break
        if tags[k.n0ledge + i] == tags[k.s0ledge + i]:
            context[pcopy] += 1
        else:
            context[pexact] = 0
            break
        

cdef int free_predicate(Predicate* pred) except -1:
    free(pred.raws)
    free(pred.args)
    free(pred)
 

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
        self.features = <uint64_t*>calloc(self.n + (self.nr_match*2) + 1, sizeof(uint64_t))

    def __dealloc__(self):
        free(self.context)
        free(self.features)
        for i in range(self.n):
            free_predicate(self.predicates[i])
        free(self.predicates)

    cdef uint64_t* extract(self, Sentence* sent, Kernel* k) except NULL:
        cdef:
            size_t i, j, f, size
            uint64_t value
            bint seen_non_zero, seen_masked
            Predicate* pred

        cdef size_t* context = self.context
        cdef uint64_t* features = self.features
        fill_context(context, self.nr_label, sent.words, sent.pos,
                     sent.clusters, sent.cprefix4s, sent.cprefix6s,
                     sent.orths, sent.parens, sent.quotes,
                     k, &k.s0l, &k.s0r, &k.n0l)
        f = 0
        for i in range(self.n):
            pred = self.predicates[i]
            seen_non_zero = False
            seen_masked = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                #if value == self.mask_value:
                #    seen_masked = True
                #    break
                if value != 0:
                    seen_non_zero = True
                pred.raws[j] = value
            if seen_non_zero and not seen_masked:
                pred.raws[pred.n] = pred.id
                size = (pred.n + 1) * sizeof(uint64_t)
                features[f] = MurmurHash64A(pred.raws, size, i)
                f += 1
        cdef MatchPred* match_pred
        cdef size_t match_id
        for match_id in range(self.nr_match):
            match_pred = self.match_preds[match_id]
            value = context[match_pred.idx1]
            if value != 0 and value == context[match_pred.idx2]:
                match_pred.raws[0] = value
                match_pred.raws[1] = match_pred.id
                features[f] = MurmurHash64A(match_pred.raws, 2 * sizeof(size_t), match_pred.id)
                f += 1
                match_pred.raws[0] = 0
                features[f] = MurmurHash64A(match_pred.raws, 2 * sizeof(size_t), match_pred.id)
                f += 1
        features[f] = 0
        return features

    def _make_predicates(self, object name, object ngrams, add_clusters):
        feats, match_feats = self._get_feats(name, ngrams, add_clusters)
        self.n = len(feats)
        self.predicates = <Predicate**>malloc(self.n * sizeof(Predicate*))
        cdef Predicate* pred
        for id_, args in enumerate(feats):
            pred = <Predicate*>malloc(sizeof(Predicate))
            pred.id = id_
            pred.n = len(args)
            pred.expected_size = 1000
            pred.raws = <uint64_t*>malloc((len(args) + 1) * sizeof(uint64_t))
            pred.args = <int*>malloc(len(args) * sizeof(int))
            for i, element in enumerate(sorted(args)):
                pred.args[i] = element
            self.predicates[id_] = pred
        self.nr_match = len(match_feats)
        self.match_preds = <MatchPred**>malloc(len(match_feats) * sizeof(MatchPred))
        cdef MatchPred* match_pred
        for id_, (idx1, idx2) in enumerate(match_feats):
            match_pred = <MatchPred*>malloc(sizeof(MatchPred))
            match_pred.id = id_ + self.n
            match_pred.idx1 = idx1
            match_pred.idx2 = idx2
            self.match_preds[id_] = match_pred

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
            (N0orth,),
            (N1orth,),
            (N0paren, N0w),
            (N0paren, S0w),
            (N0quote,),
            (N0quote, N0w),
            (N0quote, S0w),
            (S0rw, N0orth),
            (S0rw, N0orth, N1orth),
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
            #+ unigram(S0re_w, add_clusters)
            #+ unigram(N0le_w, add_clusters)
            #+ unigram(N3w, add_clusters)
            #+ unigram(S0l0w, add_clusters)
            #+ unigram(S0r0w, add_clusters)
        )

        disfl = (
            (prev_edit,),
            (prev_prev_edit,),
            (prev_edit_wmatch,),
            (prev_edit_pmatch,),
            (prev_edit_word,),
            (prev_edit_pos,),
            (wcopy,),
            (pcopy,),
            (wexact,),
            (pexact,),
            (wcopy, pcopy),
            (wexact, pexact),
            (wexact, pcopy),
            (wcopy, pexact)
        )
        print "Use Zhang feats"
        feats = from_single + from_word_pairs + from_three_words + distance
        feats += valency + zhang_unigrams + third_order
        feats += labels
        feats += label_sets
        print "Using disfl feats"
        feats += disfl
        match_feats = []
        kernel_tokens = get_kernel_tokens()
        for w1, w2 in combinations(kernel_tokens, 2):
            # Words match
            match_feats.append((w1, w2))
            # POS match
            match_feats.append((w1 + 1, w2 + 1))
        print "Use %d ngram feats and %d match feats" % (len(ngrams), len(match_feats))
        if ngrams:
            feats += tuple(unigrams)
            for ngram_feat in ngrams:
                if len(ngram_feat) == 2:
                    if add_clusters:
                        feats += bigram_with_clusters(*ngram_feat)
                    else:
                        feats += bigram_no_clusters(*ngram_feat)
                elif len(ngram_feat) == 3:
                    if add_clusters:
                        feats += trigram_with_clusters(*ngram_feat)
                    else:
                        feats += trigram_no_clusters(*ngram_feat)
                else:
                    raise StandardError, ngram_feat
        # Sort each feature, and sort and unique the set of them
        return tuple(sorted(set([tuple(sorted(f)) for f in feats]))), match_feats
