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

    S0w
    S0p
    S0c
    S0c6
    S0c4
    S0L
    S0lv
    S0rv
    
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

    P3w
    P3p
    P3c
    P3c6
    P3c4
    P3L
    P3lv
    P3rv
 
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

    # Probability features
    # p=1.0 gets its own bucket
    prob1
    # These fire progressively, so if the log probability is < -1, only lp1 fires,
    # if it's -2, both lp1 and lp2 fire, but not lp3, etc
    lp1
    lp2
    lp3
    lp4
    lp5
    lp6
    lp7
    lp8
    lp9
    lp10

    CONTEXT_SIZE


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
    # print "{0:b}".format(first_4_bits).ljust(4, '0')
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
    fill_token(context, S0w, t.s0)
    fill_token(context, S0r2w, t.s0r2)
    fill_token(context, S0rw, t.s0r)
    fill_token(context, S0re_w, t.s0re)
    fill_token(context, N0le_w, t.n0le)
    fill_token(context, N0lw, t.n0l)
    fill_token(context, N0l2w, t.n0l2)
    fill_token(context, P3w, t.p2)
    fill_token(context, P2w, t.p2)
    fill_token(context, P1w, t.p1)
    fill_token(context, N0w, t.n0)
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

    cdef size_t lp_feat
    if t.n0_prob == 1:
        context[prob1] = 1
    else:
        for lp_feat in range(-t.n0_prob):
            context[lp1 + lp_feat] = 1


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
   
   # From three words
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
   (S0w, S0lL, S0l2L),
   (S0p, S0rL, S0r2L),
   (S0p, S0lL, S0l2L),
   (S0p, S0rL, S0r2L),
   (N0w, N0lL, N0l2L),
   (N0p, N0lL, N0l2L),

    # extra_labels = (
    (S0p, S0lL, S0lp),
    (S0p, S0lL, S0l2L),
    (S0p, S0rL, S0rp),
    (S0p, S0rL, S0r2L),
    (S0p, S0lL, S0rL),
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


disfl = (
    (prev_edit,),
    (prev_prev_edit,),
    (prev_edit_wmatch,),
    (prev_edit_pmatch,),
    (prev_edit_word,),
    (prev_edit_pos,),
    (next_edit,),
    (next_next_edit,),
    (next_edit_wmatch,),
    (next_edit_pmatch,),
    (next_edit_word,),
    (next_edit_pos,),
)


string_probs = (
    (prob1,),
    (lp1,),
    (lp2,),
    (lp3,),
    (lp4,),
    (lp5,),
    (lp6,),
    (lp7,),
    (lp8,),
    (lp9,),
    (lp10,)
)


bigrams = (
    (P1p, N0p),
    (P1p, N0w),
    (P1w, N0p),
    (P1w, N0w),
)

trigrams = (
    (P2p, P1p, N0p),
    (P2p, P1p, N0w),
)
