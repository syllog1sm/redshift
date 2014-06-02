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

    S1rw
    S1rp
    S1rc
    S1rc6
    S1rc4
    S1rL
    S1rlv
    S1rrv

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
    
    S0nw
    S0np
    S0nc
    S0nc6
    S0nc4
    S0nL
    S0nlv
    S0nrv

    S0nnw
    S0nnp
    S0nnc
    S0nnc6
    S0nnc4
    S0nnL
    S0nnlv
    S0nnrv

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

    P2w
    P2p
    P2c
    P2c6
    P2c4
    P2L
    P2lv
    P2rv
 
    P1w
    P1p
    P1c
    P1c6
    P1c4
    P1L
    P1lv
    P1rv
 
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
    # We've read in the string little-endian, so now we can take & (2**n)-1
    # to get the first n bits of the cluster.
    # e.g. s = "1110010101"
    # s = ''.join(reversed(s))
    # first_4_bits = int(s, 2)
    # print first_4_bits
    # 5
    # print "{0:b}".format(prefix).ljust(4, '0')
    # 1110
    # What we're doing here is picking a number where all bits are 1, e.g.
    # 15 is 1111, 63 is 111111 and doing bitwise AND, so getting all bits in
    # the source that are set to 1.
    context[i+2] = word.cluster
    context[i+3] = word.cluster & 63
    context[i+4] = word.cluster & 15
    context[i+5] = token.label
    context[i+6] = token.l_valency
    context[i+7] = token.r_valency

cdef inline void zero_token(size_t* context, size_t i):
    cdef size_t j
    for j in range(9):
        context[i+j] = 0


cdef int fill_context(size_t* context, SlotTokens* t) except -1:
    cdef size_t c
    for c in range(CONTEXT_SIZE):
        context[c] = 0
    # This fills in the basic properties of each of our "slot" tokens, e.g.
    # word on top of the stack, word at the front of the buffer, etc.
    fill_token(context, S2w, t.s2)
    fill_token(context, S1w, t.s1)
    fill_token(context, S1rw, t.s1r)
    fill_token(context, S0le_w, t.s0le)
    fill_token(context, S0lw, t.s0l)
    fill_token(context, S0l2w, t.s0l2)
    fill_token(context, S0l0w, t.s0l0)
    fill_token(context, S0w, t.s0)
    fill_token(context, S0nw, t.s0n)
    fill_token(context, S0nnw, t.s0nn)
    fill_token(context, S0r0w, t.s0r0)
    fill_token(context, S0r2w, t.s0r2)
    fill_token(context, S0rw, t.s0r)
    fill_token(context, S0re_w, t.s0re)
    fill_token(context, N0le_w, t.n0le)
    fill_token(context, N0lw, t.n0l)
    fill_token(context, N0l2w, t.n0l2)
    fill_token(context, N0l0w, t.n0l0)
    fill_token(context, P2w, t.p2)
    fill_token(context, P1w, t.p1)
    fill_token(context, N0w, t.n0)
    fill_token(context, N1w, t.n1)
    fill_token(context, N2w, t.n2)

    # TODO: Distance
    if t.s0.i != 0:
        assert t.n0.i > t.s0.i
        context[dist] = t.n0.i - t.s0.i
    else:
        context[dist] = 0
    # Disfluency match features
    context[prev_edit] = t.p1.is_edit
    context[prev_edit_wmatch] = t.p1.is_edit and t.p1.word == t.n0.word
    context[prev_edit_pmatch] = t.p1.is_edit and t.p1.tag == t.n0.tag
    context[prev_prev_edit] = t.p1.is_edit and t.p2.is_edit
    context[prev_edit_word] = t.p1.word.norm if t.p1.is_edit else 0
    context[prev_edit_pos] = t.p1.tag if t.p1.is_edit else 0
    
    
    context[next_edit] = t.s0n.is_edit
    context[next_edit_wmatch] = t.s0n.is_edit and t.s0n.word == t.s0.word
    context[next_edit_pmatch] = t.s0n.is_edit and t.s0n.tag == t.s0.tag
    context[next_next_edit] = t.s0n.is_edit and t.s0nn.is_edit
    context[next_edit_word] = t.s0n.is_edit and t.s0n.word.norm
    context[next_edit_pos] = t.s0n.is_edit and t.s0n.tag

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
    cdef size_t n0ledge = t.n0.left_edge
    cdef size_t s0ledge = t.s0.left_edge
    #for i in range(5):
    #    if ((n0ledge + i) > t.n0.i) or ((s0ledge + i) > t.s0.i):
    #        break
    #    if context[wexact]:
    #        if parse[n0ledge + i].word.orig == parse[s0ledge + i].word.orig:
    #            context[wcopy] += 1
    #        else:
    #            context[wexact] = 0
    #    if context[pexact]:
    #        if parse[n0ledge + i].tag == parse[s0ledge + i].tag:
    #            context[pcopy] += 1
    #        else:
    #            context[pexact] = 0
    #    if context[wsexact]:
    #        if parse[t.s0.i - i].word.orig == parse[t.n0.i - i].word.orig:
    #            context[wscopy] += 1
    #        else:
    #            context[wsexact] = 0
    #    if context[psexact]:
    #        if parse[t.s0.i - i].tag == parse[t.n0.i - i].tag:
    #            context[pscopy] += 1
    #        else:
    #            context[psexact] = 0


arc_hybrid = (
    # Single words
    (S2w,),
    (S1w,),
    (S0lw,),
    (S0l2w,),
    (S0w,),
    (S0r2w,),
    (S0rw,),
    (N0lw,),
    (N0l2w,),
    (N0w,),
    (N1w,),
    (N2w,),

    # Single tags
    (S2p,),
    (S1p,),
    (S0lp,),
    (S0l2p,),
    (S0p,),
    (S0r2p,),
    (S0rp,),
    (N0lp,),
    (N0l2p,),
    (N0p,),
    (N1p,),
    (N2p,),

    # Single word + single tag
    (S2w, S2p,),
    (S1w, S1p,),
    (S0lw, S0lp,),
    (S0l2w, S0l2p,),
    (S0w, S0p,),
    (S0r2w, S0r2p,),
    (S0rw, S0rp,),
    (N0lw, N0lp,),
    (N0l2w, N0l2p,),
    (N0w, N0p,),
    (N1w, N1p,),
    (N2w, N2p,),

   # Word pairs 
   # S0, N0
   (S0w, S0p, N0w, N0p),
   (S0w, S0p, N0w),
   (S0w, N0w, N0p),
   (S0w, S0p, N0p),
   (S0p, N0w, N0p),
   (S0w, N0w),
   (S0p, N0p),

   # S1, S0
   (S1w, S1p, S0w, S0p),
   (S1w, S1p, S0w),
   (S1w, S0w, S0p),
   (S1w, S1p, S0p),
   (S1p, S0w, S0p),
   (S1p, S0p),
   
   (N0p, N1p),

   # From three words
   (N0p, N1p, N2p),
   (S0p, N0p, N1p),
   (S1p, S0p, N0p),
   (S0p, S0lp, N0p),
   (S0p, S0rp, N0p),
   (S0p, N0p, N0lp),

   # Distance
   (dist, S0w),
   (dist, S0p),
   (dist, N0w),
   (dist, N0p),
   (dist, S0w, N0w),
   (dist, S0p, N0p),
   
   # Valency
   (S0w, S0rv),
   (S0p, S0rv),
   (S0w, S0lv),
   (S0p, S0lv),
   (N0w, N0lv),
   (N0p, N0lv),
   
   # Third order
   (S0p, S0lp, S0l2p),
   (S0p, S0rp, S0r2p),
   (S0p, S1p, S2p),
   (N0p, N0lp, N0l2p),
   
   # Labels
   (S0lL,),
   (S0rL,),
   (N0lL,),
   (S0l2L,),
   (S0r2L,),
   (N0l2L,),

   # Label sets
   (S0w, S0lL, S0l0L, S0l2L),
   (S0p, S0rL, S0r0L, S0r2L),
   (S0p, S0lL, S0l0L, S0l2L),
   (S0p, S0rL, S0r0L, S0r2L),
   (N0w, N0lL, N0l0L, N0l2L),
   (N0p, N0lL, N0l0L, N0l2L),

   # Stack-second
    #(S1w, N0w),
    #(S1w, N0p),
    #(S1p, N0w),
    #(S1p, N0p),
    #(S1w, N1w),
    #(S1w, N1p),
    #(S1p, N1p),
    #(S1p, N1w),
    #(S1p, S0p, N0p),
    #(S1w, S0w, N0w),
    #(S1w, S0p, N0p),
    #(S2w, N0w),
    #(S2w, N1w),
    #(S2p, N0p, N1w),
    #(S2p, N0w, N1w),
    #(S2w, N0p, N1p),

    # S1r
    #(S1rp, S0p),
    #(S1rp, S0w),
    #(S1rw, S0p),

    #(S1p, S1rL),
    #(S1w, S1rL),
    #(S1rp, S1rL),
    #(S1rw, S1rL),
    #(S1rL, S0w),
    #(S1rL, S0p),
    #(S1p, S1rL, S0p),
    #(S1rv, S0p),
    #(S1rv, S0w),

    # For Break
    #(S0re_p, N0p),
    #(S0re_w, N0p),
    #(S0re_p, N0w),

    #(S0re_p, N0le_p),
    #(S0re_w, N0le_p),
    #(S0re_p, N0le_w)
)

extra_labels = (
    (S0p, S0lL, S0lp),
    (S0p, S0lL, S0l2L),
    (S0p, S0rL, S0rp),
    (S0p, S0rL, S0r2L),
    (S0p, S0lL, S0rL),
    (S0p, S0lL, S0l2L, S0l0L),
    (S0p, S0rL, S0r2L, S0r0L),
    (S1p, S0L, S0rL),
    (S1p, S0L, S0lL),
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
    (S1p, S0L, S0rL),
    (S1p, S0L, S0lL),
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
    (S1c4, S0c4, N0c4),
    (S1c6, S0c6, N0c6),
    (S1p, S0c4, N0c4),
    (S1c4, S0p, N0c4),
    (S1c4, S0c4, N0p),
    # Grand-child, left-arc
    (S0lc4, S0c4, N0c4),
    (S0lc6, S0c6, N0c6),
    (S0lp, S0c4, N0c4),
    (S0lc4, S0p, N0c4),
    (S0lc4, S0c4, N0p)
)

prev_next = (
    (P2w,),
    (P2p,),
    (P1w,),
    (P1p,),
    (P2p, P1p,),
    (P1p, N0w,),
    (P1p, N0p,),

    (S0nw,),
    (S0np,),
    (S0nnw,),
    (S0nnp,),
    (S0nnp, S0np),
    (S0np, S0w,),
    (S0np, S0p,),

    (S0np, N0w),
    (S0np, N0p),

    (S0w, P1p),
    (S0p, P1p),
    (S0p, P1w),
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


def match_templates():
    match_feats = []
    kernel_tokens = get_kernel_tokens()
    for w1, w2 in combinations(kernel_tokens, 2):
        # Words match
        match_feats.append((w1, w2))
        # POS match
        match_feats.append((w1 + 1, w2 + 1))
    return tuple(match_feats)
