# cython: profile=True
"""
Fill an array, context, with every _atomic_ value our features reference.
We then write the _actual features_ as tuples of the atoms. The machinery
that translates from the tuples to feature-extractors (which pick the values
out of "context") is in features/extractor.pyx

The atomic feature names are listed in a big enum, so that the feature tuples
can refer to them.
"""

from redshift._state cimport SlotTokens
from redshift.sentence cimport Token
from index.lexicon cimport Lexeme
from itertools import combinations
# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum
# Ensure each token's attributes are listed: w, p, c, c6, c4. The order
# is referenced by incrementing the enum...
# Tokens are listed in left-to-right order.
#cdef size_t* SLOTS = [
#    S2w, S1w,
#    S0l0w, S0l2w, S0lw,
#    S0w,
#    S0r0w, S0r2w, S0rw,
#    N0l0w, N0l2w, N0lw,
#    N0w, N1w, N2w, N3w, 0
#]
# NB: The order of the enum is _not arbitrary!!_
cdef enum:
    S2w
    S2p
    S2c
    S2c6
    S2c4
    S2L
    S2lv
    S2rv

    S1w
    S1p
    S1c
    S1c6
    S1c4
    S1L
    S1lv
    S1rv

    S0lw
    S0lp
    S0lc
    S0lc6
    S0lc4
    S0lL
    S0llv
    S0lrv

    S0l2w
    S0l2p
    S0l2c
    S0l2c6
    S0l2c4
    S0l2L
    S0l2lv
    S0l2rv

    S0l0w
    S0l0p
    S0l0c
    S0l0c6
    S0l0c4
    S0l0L
    S0l0lv
    S0l0rv

    S0w
    S0p
    S0c
    S0c6
    S0c4
    S0L
    S0lv
    S0rv

    S0r0w
    S0r0p
    S0r0c
    S0r0c6
    S0r0c4
    S0r0L
    S0r0lv
    S0r0rv

    S0r2w
    S0r2p
    S0r2c
    S0r2c6
    S0r2c4
    S0r2L
    S0r2lv
    S0r2rv

    S0rw
    S0rp
    S0rc
    S0rc6
    S0rc4
    S0rL
    S0rlv
    S0rrv

    N0l0w
    N0l0p
    N0l0c
    N0l0c6
    N0l0c4
    N0l0L
    N0l0lv
    N0l0rv

    N0l2w
    N0l2p
    N0l2c
    N0l2c6
    N0l2c4
    N0l2L
    N0l2lv
    N0l2rv

    N0lw
    N0lp
    N0lc
    N0lc6
    N0lc4
    N0lL
    N0llv
    N0lrv

    N0w
    N0p
    N0c
    N0c6
    N0c4
    N0L
    N0lv
    N0rv
 
    N1w
    N1p
    N1c
    N1c6
    N1c4
    N1L
    N1lv
    N1rv
    
    N2w
    N2p
    N2c
    N2c6
    N2c4
    N2L
    N2lv
    N2rv
    
    N3w
    N3p
    N3c
    N3c6
    N3c4
    N3L
    N3lv
    N3rv

    S0hw
    S0hp
    S0hc
    S0hc6
    S0hc4
    S0hL
    S0hlv
    S0hrv
 
    S0h2w
    S0h2p
    S0h2c
    S0h2c6
    S0h2c4
    S0h2L
    S0h2lv
    S0h2rv
    
    S0le_w
    S0le_p
    S0le_c
    S0le_c6
    S0le_c4
    S0le_L
    S0le_lv
    S0le_rv
    
    S0re_w
    S0re_p
    S0re_c
    S0re_c6
    S0re_c4
    S0re_L
    S0re_lv
    S0re_rv
    
    N0le_w
    N0le_p
    N0le_c
    N0le_c6
    N0le_c4
    N0le_L
    N0le_lv
    N0le_rv

    dist
    
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

    wcopy 
    pcopy

    wexact
    pexact

    wscopy
    pscopy

    wsexact
    psexact

    CONTEXT_SIZE


# Listed in left-to-right order
cdef size_t[16] SLOTS
SLOTS[0] = S2w; SLOTS[1] = S1w
SLOTS[2] = S0le_w; SLOTS[3] = S0lw; SLOTS[4] = S0l2w; SLOTS[5] = S0l0w
SLOTS[6] = S0w
SLOTS[7] = S0r0w; SLOTS[8] = S0r2w; SLOTS[9] = S0rw; SLOTS[10] = S0re_w
SLOTS[11] = N0le_w; SLOTS[12] = N0l0w; SLOTS[13] = N0l2w; SLOTS[14] = N0lw
SLOTS[15] = N0w


cdef size_t NR_SLOT = sizeof(SLOTS) / sizeof(SLOTS[0])


def get_kernel_tokens():
    kernel = []
    for i in range(NR_SLOT):
        kernel.append(SLOTS[i])
    return kernel

def context_size():
    return CONTEXT_SIZE

cdef inline void fill_token(size_t* context, size_t i, Token token):
    cdef Lexeme* word = token.word
    context[i] = word.norm
    context[i+1] = token.tag
    # TODO: Implement 4 and 6 bit cluster prefixes
    context[i+2] = word.cluster
    context[i+3] = word.cluster
    context[i+4] = word.cluster
    context[i+5] = token.label
    context[i+6] = token.l_valency
    context[i+7] = token.r_valency

cdef inline void zero_token(size_t* context, size_t i):
    cdef size_t j
    for j in range(9):
        context[i+j] = 0


cdef int fill_context(size_t* context, SlotTokens* tokens, Token* parse,
                      Step* steps) except -1:
    cdef size_t c
    for c in range(CONTEXT_SIZE):
        context[c] = 0
    # This fills in the basic properties of each of our "slot" tokens, e.g.
    # word on top of the stack, word at the front of the buffer, etc.
    fill_token(context, S2w, tokens.s2)
    fill_token(context, S1w, tokens.s1)
    fill_token(context, S0le_w, tokens.s0le)
    fill_token(context, S0lw, tokens.s0l)
    fill_token(context, S0l2w, tokens.s0l2)
    fill_token(context, S0l0w, tokens.s0l0)
    fill_token(context, S0w, tokens.s0)
    fill_token(context, S0r0w, tokens.s0r0)
    fill_token(context, S0r2w, tokens.s0r2)
    fill_token(context, S0rw, tokens.s0r)
    fill_token(context, S0re_w, tokens.s0re)
    fill_token(context, N0le_w, tokens.n0le)
    fill_token(context, N0lw, tokens.n0l)
    fill_token(context, N0l2w, tokens.n0l2)
    fill_token(context, N0l0w, tokens.n0l0)
    fill_token(context, N0w, tokens.n0)
    fill_token(context, N1w, tokens.n1)
    fill_token(context, N2w, tokens.n2)

    if tokens.s0.label != 0:
        fill_token(context, S0hw, tokens.s1)
    else:
        zero_token(context, S0hw)
    if tokens.s0.label != 0 and tokens.s1.label != 0:
        fill_token(context, S0h2w, tokens.s2)
    else:
        zero_token(context, S0h2w)
    # TODO: Distance
    #if tokens.s0.word.orig != 0:
    #    assert tokens.n0 > tokens.s0
    #    context[dist] = tokens.n0 - tokens.s0
    #else:
    #    context[dist] = 0
    """
    # Disfluency match features
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
    if k.next_edit and k.s0 != 0:
        context[next_edit] = 1
        context[next_edit_wmatch] = 1 if words[k.s0 + 1] == words[k.s0] else 0
        context[next_edit_pmatch] = 1 if tags[k.s0 + 1] == tags[k.s0] else 0
        context[next_next_edit] = 1 if k.next_next_edit else 0
        context[next_edit_word] = words[k.s0 + 1]
        context[next_edit_pos] = k.next_tag
    else:
        context[next_edit] = 0
        context[next_edit_wmatch] = 0
        context[next_edit_pmatch] = 0
        context[next_next_edit] = 0
        context[next_edit_word] = 0
        context[next_edit_pos] = 0

    # These features find how much of S0's span matches N0's span, starting from
    # the left.
    # 
    context[wcopy] = 0
    context[wexact] = 1
    context[pcopy] = 0
    context[pexact] = 1
    context[wscopy] = 0
    context[wsexact] = 1
    context[pscopy] = 0
    context[psexact] = 1
    for i in range(5):
        if ((k.n0ledge + i) > k.i) or ((k.s0ledge + i) > k.s0):
            break
        if context[wexact]:
            if words[k.n0ledge + i] == words[k.s0ledge + i]:
                context[wcopy] += 1
            else:
                context[wexact] = 0
        if context[pexact]:
            if tags[k.n0ledge + i] == tags[k.s0ledge + i]:
                context[pcopy] += 1
            else:
                context[pexact] = 0
        if context[wsexact]:
            if words[k.s0 - i] == words[k.i - i]:
                context[wscopy] += 1
            else:
                context[wsexact] = 0
        if context[psexact]:
            if tags[k.s0 - i] == tags[k.i - i]:
                context[pscopy] += 1
            else:
                context[psexact] = 0
    """


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
   (S0L,),
   (S0lL,),
   (S0rL,),
   (N0lL,),
   (S0hL,),
   (S0l2L,),
   (S0r2L,),
   (N0l2L,),
)


label_sets = (
   (S0w, S0lL, S0l0L, S0l2L),
   (S0p, S0rL, S0r0L, S0r2L),
   (S0p, S0lL, S0l0L, S0l2L),
   (S0p, S0rL, S0r0L, S0r2L),
   (N0w, N0lL, N0l0L, N0l2L),
   (N0p, N0lL, N0l0L, N0l2L),
)

extra_labels = (
    (S0p, S0lL, S0lp),
    (S0p, S0lL, S0l2L),
    (S0p, S0rL, S0rp),
    (S0p, S0rL, S0r2L),
    (S0p, S0lL, S0rL),
    (S0p, S0lL, S0l2L, S0l0L),
    (S0p, S0rL, S0r2L, S0r0L),
    (S0hp, S0L, S0rL),
    (S0hp, S0L, S0lL),
)

edges = (
    (S0re_w,),
    (S0re_p,),
    (S0re_w, S0re_p),
    (S0le_w,),
    (S0le_p,),
    (S0le_w, S0le_p),
    (N0le_w,),
    (N0le_p,),
    (N0le_w, N0le_p),
    (S0re_p, N0p,),
    (S0p, N0le_p)
)


stack_second = (
    (S1w,),
    (S1p,),
    (S1w, S1p),
    (S1w, N0w),
    (S1w, N0p),
    (S1p, N0w),
    (S1p, N0p),
    #(S1w, N1w),
    (S1w, N1p),
    (S1p, N1p),
    (S1p, N1w),
    (S1p, S0p, N0p),
    #(S1w, S0w, N0w),
    (S1w, S0p, N0p),
    #(S2w, N0w),
    #(S2w, N1w),
    (S2p, N0p, N1w),
    #(S2p, N0w, N1w),
    (S2w, N0p, N1p),
)

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


def pos_bigrams():
    kernels = [S2w, S1w, S0w, S0lw, S0rw, N0w, N0lw, N1w]
    bitags = []
    for t1, t2 in combinations(kernels, 2):
        feat = (t1 + 1, t2 + 1)
        bitags.append(feat)
    print "Adding %d bitags" % len(bitags)
    return tuple(bitags)


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
