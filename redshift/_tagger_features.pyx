from index.lexicon cimport Lexeme

cdef enum:
    P1p
    P2p

    N0w
    N0c
    N0c6
    N0c4
    N0pre
    N0suff
    N0title
    N0upper
    N0alpha

    N1w
    N1c
    N1c6
    N1c4
    N1pre
    N1suff
    N1title
    N1upper
    N1alpha

    N2w
    N2c
    N2c6
    N2c4
    N2pre
    N2suff
    N2title
    N2upper
    N2alpha

    N3w
    N3c
    N3c6
    N3c4
    N3pre
    N3suff
    N3title
    N3upper
    N3alpha

    P1w
    P1c
    P1c6
    P1c4
    P1pre
    P1suff
    P1title
    P1upper
    P1alpha

    P2w
    P2c
    P2c6
    P2c4
    P2pre
    P2suff
    P2title
    P2upper
    P2alpha

    CONTEXT_SIZE

def context_size():
    return CONTEXT_SIZE


cdef int fill_context(atom_t* context, Sentence* sent, size_t ptag, size_t pptag,
                      size_t i):
    for j in range(CONTEXT_SIZE):
        context[j] = 0
    context[P1p] = ptag
    context[P2p] = pptag
    
    fill_token(context, N0w, sent.tokens[i].word)
    fill_token(context, N1w, sent.tokens[i+1].word)
    if (i + 2) < sent.n:
        fill_token(context, N2w, sent.tokens[i+2].word)
    if (i + 3) < sent.n:
        fill_token(context, N3w, sent.tokens[i+3].word)
    if i >= 1:
        fill_token(context, P1w, sent.tokens[i-1].word)
    if i >= 2:
        fill_token(context, P2w, sent.tokens[i-2].word)


cdef inline void fill_token(atom_t* context, size_t i, Lexeme* word):
    context[i] = word.norm
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
    context[i+1] = word.cluster
    context[i+2] = word.cluster & 63
    context[i+3] = word.cluster & 15
    context[i+4] = word.prefix
    context[i+5] = word.suffix
    context[i+6] = word.oft_title
    context[i+7] = word.oft_upper
    context[i+8] = word.non_alpha


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
    (P1p,),
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

orth = (
    (N0pre,),
    (N1pre,),
    (P1pre,),
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

feature_set = basic + clusters + case + orth

debug_features = ((N0w,), (P1p,))
