from redshift._state cimport Kernel, Subtree
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

    N0l0w
    N0l0p
    N0l0c
    N0l0c6
    N0l0c4
    
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

    S0le_w
    S0le_p
    S0le_c
    S0le_c6
    S0le_c4
    
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


def context_size():
    return CONTEXT_SIZE

def get_kernel_tokens():
    return [S0w, N0w]


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

    context[N0l0w] = words[n0l.idx[2]]
    context[N0l0p] = n0l.tags[2]
    context[N0l0c] = clusters[n0l.idx[2]]
    context[N0l0c6] = cprefix6s[n0l.idx[2]]
    context[N0l0c4] = cprefix4s[n0l.idx[2]]

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

    context[S0le_w] = words[k.s0ledge]
    context[S0le_p] = tags[k.s0ledge]
    context[S0le_c] = clusters[k.s0ledge]
    context[S0le_c6] = cprefix6s[k.s0ledge]
    context[S0le_c4] = cprefix4s[k.s0ledge]
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
   (S0hw, N0w),
   (S0hp, N0p),
   (S0hw, N1w),
   (S0hp, N1w),
   (S0hp, S0p, N0p)
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
    (wcopy, pexact),
    (prev_edit, wcopy),
    (prev_prev_edit, wcopy),
    (prev_edit, pcopy),
    (prev_prev_edit, pcopy)
)

def get_best_bigrams(all_bigrams, n=0):
    return []

def get_best_trigrams(all_trigrams, n=0):
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


def bigram(a, b, add_clusters=False):
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


def trigram(a, b, c, add_clusters=False):
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


  
def baseline_templates():
    return from_single + from_word_pairs + from_three_words + distance + \
           valency + zhang_unigrams + third_order + labels + label_sets


def match_templates():
    match_feats = []
    kernel_tokens = get_kernel_tokens()
    for w1, w2 in combinations(kernel_tokens, 2):
        # Words match
        match_feats.append((w1, w2))
        # POS match
        match_feats.append((w1 + 1, w2 + 1))
    return tuple(match_feats)


def ngram_feats(ngrams, add_clusters=False):
    kernel_tokens = get_kernel_tokens()
    feats = []
    for token in kernel_tokens:
        feats.extend(unigram(token, add_clusters=add_clusters))
    for ngram_feat in ngrams:
        if len(ngram_feat) == 2:
            feats += bigram(*ngram_feat, add_clusters=add_clusters)
        elif len(ngram_feat) == 3:
            feats += trigram(*ngram_feat, add_clusters=add_clusters)
        else:
            raise StandardError, ngram_feat
    return tuple(feats)
