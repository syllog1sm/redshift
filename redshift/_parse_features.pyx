from redshift._fast_state cimport Kernel, Subtree
from itertools import combinations

# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum
# Ensure each token's attributes are listed: w, p, c, cp

cdef enum:
    S0w
    S0p
    S0c
    S0c6
    S0c4
    
    S0lw
    S0lp
    S0lc
    S0lc6
    S0lc4

    S0l2w
    S0l2p
    S0l2c
    S0l2c6
    S0l2c4

    S0lew
    S0lep
    S0lec
    S0lec6
    S0lec4

    S0l0w
    S0l0p
    S0l0c
    S0l0c6
    S0l0c4
    
    S0rw
    S0rp
    S0rc
    S0rc6
    S0rc4

    S0r2w
    S0r2p
    S0r2c
    S0r2c6
    S0r2c4

    S0rew
    S0rep
    S0rec
    S0rec6
    S0rec4

    S0r0w
    S0r0p
    S0r0c
    S0r0c6
    S0r0c4
    
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

    N0l2w
    N0l2p
    N0l2c
    N0l2c6
    N0l2c4

    N0lew
    N0lep
    N0lec
    N0lec6
    N0lec4

    N0l0w
    N0l0p
    N0l0c
    N0l0c6
    N0l0c4
   
    S0hw
    S0hp
    S0hc
    S0hc6
    S0hc4

    S0h2w
    S0h2p
    S0h2c
    S0h2c6
    S0h2c4

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
    
    S0l
    S0ll
    S0l2l
    S0rl
    S0r2l

    N0ll
    N0l2l

    S0hl
    S0h2l
    
    N0lv
    S0lv
    S0rv
    dist

    wcopy
    pcopy
    wexact
    pexact

    prev_edit
    prev_edit_wmatch
    prev_edit_pmatch
    prev_edit_word
    prev_edit_pos
    prev_prev_edit

    next_edit
    next_edit_wmatch
    next_edit_pmatch
    next_edit_word
    next_edit_pos
    next_next_edit

    CONTEXT_SIZE


def context_size():
    return CONTEXT_SIZE


def get_kernel_tokens():
    return [S0hw, S0h2w, S0w, S0lw, S0l2w, S0rw, S0r2w,
            N0w, N0lw, N0l2w, N1w, N2w, S0lew, N0lew, S0rew, S0l0w, N0l0w, S0r0w]


cdef void fill_context(size_t* context, size_t nr_label, size_t* words,
                       size_t* tags,
                       size_t* clusters, size_t* cprefix6s, size_t* cprefix4s,
                       size_t* orths, int* parens, int* quotes,
                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l):

    context[S0w] = words[k.s0]
    context[S0p] = tags[k.s0]
    context[S0c] = clusters[k.s0]
    context[S0c6] = cprefix6s[k.s0]
    context[S0c4] = cprefix4s[k.s0]

    # TODO: Bug with these values not being zeroed on 0-valencies. Probably
    # messes up word match features.
    context[S0lw] = words[s0l.kids[0].idx]
    context[S0lp] = tags[s0l.kids[0].idx]
    context[S0lc] = clusters[s0l.kids[0].idx]
    context[S0lc6] = cprefix6s[s0l.kids[0].idx]
    context[S0lc4] = cprefix4s[s0l.kids[0].idx]

    context[S0l2w] = words[s0l.kids[1].idx]
    context[S0l2p] = tags[s0l.kids[1].idx]
    context[S0l2c] = clusters[s0l.kids[1].idx]
    context[S0l2c6] = cprefix6s[s0l.kids[1].idx]
    context[S0l2c4] = cprefix4s[s0l.kids[1].idx]

    context[S0lew] = words[s0l.edge]
    context[S0lep] = tags[s0l.edge]
    context[S0lec] = clusters[s0l.edge]
    context[S0lec6] = cprefix6s[s0l.edge]
    context[S0lec4] = cprefix4s[s0l.edge]
    
    context[S0l0w] = words[s0l.first]
    context[S0l0p] = tags[s0l.first]
    context[S0l0c] = clusters[s0l.first]
    context[S0l0c6] = cprefix6s[s0l.first]
    context[S0l0c4] = cprefix4s[s0l.first]

    context[S0rw] = words[s0r.kids[0].idx]
    context[S0rp] = tags[s0r.kids[0].idx]
    context[S0rc] = clusters[s0r.kids[0].idx]
    context[S0rc6] = cprefix6s[s0r.kids[0].idx]
    context[S0rc4] = cprefix4s[s0r.kids[0].idx]
    
    context[S0r2w] = words[s0r.kids[1].idx]
    context[S0r2p] = tags[s0r.kids[1].idx]
    context[S0r2c] = clusters[s0r.kids[1].idx]
    context[S0r2c6] = cprefix6s[s0r.kids[1].idx]
    context[S0r2c4] = cprefix4s[s0r.kids[1].idx]

    context[S0rew] = words[s0r.edge]
    context[S0rep] = tags[s0r.edge]
    context[S0rec] = clusters[s0r.edge]
    context[S0rec6] = cprefix6s[s0r.edge]
    context[S0rec4] = cprefix4s[s0r.edge]
    
    context[S0r0w] = words[s0r.first]
    context[S0r0p] = tags[s0r.first]
    context[S0r0c] = clusters[s0r.first]
    context[S0r0c6] = cprefix6s[s0r.first]
    context[S0r0c4] = cprefix4s[s0r.first]

    context[N0w] = words[k.i]
    context[N0p] = tags[k.i]
    context[N0c] = clusters[k.i]
    context[N0c6] = cprefix6s[k.i]
    context[N0c4] = cprefix4s[k.i]

    context[N0lw] = words[n0l.kids[0].idx]
    context[N0lp] = tags[n0l.kids[0].idx]
    context[N0lc] = clusters[n0l.kids[0].idx]
    context[N0lc6] = cprefix6s[n0l.kids[0].idx]
    context[N0lc4] = cprefix4s[n0l.kids[0].idx]
    
    context[N0l2w] = words[n0l.kids[1].idx]
    context[N0l2p] = tags[n0l.kids[1].idx]
    context[N0l2c] = clusters[n0l.kids[1].idx]
    context[N0lc6] = cprefix6s[n0l.kids[1].idx]
    context[N0l2c4] = cprefix4s[n0l.kids[1].idx]

    context[N0lew] = words[n0l.edge]
    context[N0lep] = tags[n0l.edge]
    context[N0lec] = clusters[n0l.edge]
    context[N0lec6] = cprefix6s[n0l.edge]
    context[N0lec4] = cprefix4s[n0l.edge]
    
    context[N0l0w] = words[n0l.first]
    context[N0l0p] = tags[n0l.first]
    context[N0l0c] = clusters[n0l.first]
    context[N0l0c6] = cprefix6s[n0l.first]
    context[N0l0c4] = cprefix4s[n0l.first]

    context[N1w] = words[k.i + 1]
    context[N1p] = tags[k.i + 1]
    context[N1c] = clusters[k.i + 1]
    context[N1c6] = cprefix6s[k.i + 1]
    context[N1c4] = cprefix4s[k.i + 1]
    
    context[N2w] = words[k.i + 2]
    context[N2p] = tags[k.i + 2]
    context[N2c] = clusters[k.i + 2]
    context[N2c6] = cprefix6s[k.i + 2]
    context[N2c4] = cprefix4s[k.i + 2]
    # If there's a label set for s0, then S1 is the head of S0
    context[S0l] = k.Ls0
    if k.Ls0:
        assert k.s1
        context[S0hw] = words[k.s1]
        context[S0hp] = tags[k.s1]
        context[S0hc] = clusters[k.s1]
        context[S0hc6] = cprefix6s[k.s1]
        context[S0hc4] = cprefix4s[k.s1]
        context[S0hl] = k.Ls1
    else:
        context[S0hw] = 0
        context[S0hp] = 0
        context[S0hc] = 0
        context[S0hc6] = 0
        context[S0hc4] = 0
        context[S0hl] = 0
    # Likewise, if both S0 and S1 have labels, then S2 must be S0's grandparent
    if k.Ls0 and k.Ls1:
        assert k.s2
        context[S0h2w] = words[k.s2]
        context[S0h2p] = tags[k.s2]
        context[S0h2c] = clusters[k.s2]
        context[S0h2c6] = cprefix6s[k.s2]
        context[S0h2c4] = cprefix4s[k.s2]
        context[S0h2l] = k.Ls2
    else:
        context[S0h2w] = 0
        context[S0h2p] = 0
        context[S0h2c] = 0
        context[S0h2c6] = 0
        context[S0h2c4] = 0
        context[S0h2l] = 0

    context[S0ll] = s0l.kids[0].lab
    context[S0l2l] = s0l.kids[1].lab
    context[S0rl] = s0r.kids[0].lab
    context[S0r2l] = s0r.kids[1].lab
    context[N0ll] = n0l.kids[0].lab
    context[N0l2l] = n0l.kids[1].lab

    context[S0lv] = s0l.val
    context[S0rv] = s0r.val
    context[N0lv] = n0l.val

    # TODO: Seems hard to believe we want to keep d non-zero when there's no
    # stack top. Experiment with this futrther.
    context[wcopy] = 0
    context[wexact] = 1
    context[pcopy] = 0
    context[pexact] = 1
 
    if k.s0 != 0:
        context[dist] = k.i - k.s0
    else:
        context[dist] = 0
    for i in range(5):
        if ((n0l.edge + i) > k.i) or ((s0l.edge + i) > k.s0):
            break
        if context[wexact]:
            if words[n0l.edge + i] == words[s0l.edge + i]:
                context[wcopy] += 1
            else:
                context[wexact] = 0
        if context[pexact]:
            if tags[n0l.edge + i] == tags[s0l.edge + i]:
                context[pcopy] += 1
            else:
                context[pexact] = 0
    context[prev_edit] = k.prev_dfl != 0
    context[prev_edit_wmatch] = k.prev_dfl and words[k.prev_dfl] == words[k.s0]
    context[prev_edit_pmatch] = k.prev_dfl and tags[k.prev_dfl] == tags[k.s0]
    context[prev_edit_word] = words[k.prev_dfl] if k.prev_dfl else 0
    context[prev_edit_pos] = tags[k.prev_dfl] if k.prev_dfl else 0
    context[prev_prev_edit] = 0

    context[next_edit] = 0
    context[next_edit_wmatch] = k.next_dfl and words[k.next_dfl] == words[k.s0]
    context[next_edit_pmatch] = k.next_dfl and tags[k.next_dfl] == tags[k.s0]
    context[next_edit_word] = words[k.next_dfl] if k.next_dfl else 0
    context[next_edit_pos] = tags[k.next_dfl] if k.next_dfl else 0

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
   (S0w, S0rl, S0r2l),
   (S0p, S0rl, S0r2l),
   (S0w, S0ll, S0l2l),
   (S0p, S0ll, S0l2l),
   (N0w, N0ll, N0l2l),
   (N0p, N0ll, N0l2l),
)

extra_labels = (
    (S0p, S0ll, S0lp),
    (S0p, S0ll, S0l2l),
    (S0p, S0rl, S0rp),
    (S0p, S0rl, S0r2l),
    (S0p, S0ll, S0rl),
    (S0p, S0rl, S0r2l),
    (S0hp, S0l, S0rl),
    (S0hp, S0l, S0ll),
)

edges = (
    (S0rew,),
    (S0rep,),
    (S0rew, S0rep),
    (S0lew,),
    (S0lep,),
    (S0lew, S0lep),
    (N0lew,),
    (N0lep,),
    (N0lew, N0lep),
    (S0rep, N0p,),
    (S0p, N0lep)
)

"""
stack_second = (
    (S1w,),
    (S1p,),
    (S1w, S1p),
    (S1w, N0w),
    (S1w, N0p),
    (S1p, N0w),
    (S1p, N0p),
    (S1w, N1w),
    (S1w, N1p),
    (S1p, N1p),
    (S1p, N1w),
    (S1p, S0p, N0p),
    (S1w, S0w, N0w),
    (S1w, S0p, N0p),
    (S2w, N0w),
    (S2w, N1w),
    (S2p, N0p, N1w),
    (S2p, N0w, N1w),
    (S2w, N0p, N1p),
)
history = (
    (m1,),
    (m1, m2),
    (m1, m2, m3),
    (m1, m2, m3, m4),
    (m1, m2, m3, m4, m5),
)
"""


# Koo et al (2008) dependency features, using Brown clusters.
clusters = (
    # Koo et al have (head, child) --- we have S0, N0 for both.
    (S0c4, N0c4),
    (S0c6, N0c6),
    (S0c, N0c),
    (S0p, N0c4),
    (S0p, N0c6),
    (S0p, N0c),
    (S0c4, N0p),
    (S0c6, N0p),
    (S0c, N0p),
    # Siblings --- right arc
    (S0c4, S0rc4, N0c4),
    (S0c6, S0rc6, N0c6),
    (S0p, S0rc4, N0c4),
    (S0c4, S0rp, N0c4),
    (S0c4, S0rc4, N0p),
    # Siblings --- left arc
    (S0c4, N0lc4, N0c4),
    (S0c6, N0c6, N0c6),
    (S0c4, N0lc4, N0p),
    (S0c4, N0lp, N0c4),
    (S0p, N0lc4, N0c4),
    # Grand-child, right-arc
    (S0hc4, S0c4, N0c4),
    (S0hc6, S0c6, N0c6),
    (S0hp, S0c4, N0c4),
    (S0hc4, S0p, N0c4),
    (S0hc4, S0c4, N0p),
    # Grand-child, left-arc
    (S0lc4, S0c4, N0c4),
    (S0lc6, S0c6, N0c6),
    (S0lp, S0c4, N0c4),
    (S0lc4, S0p, N0c4),
    (S0lc4, S0c4, N0p)
)
"""

disfl = (
    #(prev_edit,),
    #(prev_prev_edit,),
    #(prev_edit_wmatch,),
    #(prev_edit_pmatch,),
    #(prev_edit_word,),
    #(prev_edit_pos,),
    (wcopy,),
    (pcopy,),
    (wexact,),
    (pexact,),
    (wcopy, pcopy),
    (wexact, pexact),
    (wexact, pcopy),
    (wcopy, pexact),
    #(prev_edit, wcopy),
    #(prev_prev_edit, wcopy),
    #(prev_edit, pcopy),
    #(prev_prev_edit, pcopy)
)


# After emailing Mark after ACL
new_disfl = (
    (next_edit,),
    (next_next_edit,),
    (next_edit_wmatch,),
    (next_edit_pmatch,),
    (next_edit_word,),
    (next_edit_pos,),
    (next_edit, wcopy),
    (next_next_edit, wcopy),
    (next_edit, pcopy),
    (next_next_edit, pcopy),
)

suffix_disfl = (
    (wscopy,),
    (pscopy,),
    (wsexact,),
    (psexact,),
    (wscopy, pscopy),
    (wsexact, psexact),
    (wsexact, pscopy),
    (wscopy, psexact),
)

def cluster_bigrams():
    kernels = [S2w, S1w, S0w, S0lw, S0rw, N0w, N0lw, N1w]
    clusters = []
    for t1, t2 in combinations(kernels, 2):
        feat = (t1 + 2, t1 + 1, t2 + 2, t2 + 1)
        clusters.append(feat)
    print "Adding %d cluster bigrams" % len(clusters)
    return tuple(clusters)


def pos_bigrams():
    kernels = [S2w, S1w, S0w, S0lw, S0rw, N0w, N0lw, N1w]
    bitags = []
    for t1, t2 in combinations(kernels, 2):
        feat = (t1 + 1, t2 + 1)
        bitags.append(feat)
    print "Adding %d bitags" % len(bitags)
    return tuple(bitags)
"""

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
    return from_single + zhang_unigrams + third_order + from_word_pairs + \
           from_three_words + labels + label_sets  + distance + valency


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


problem = (
   (S0h2p,),
)

debug = problem 
