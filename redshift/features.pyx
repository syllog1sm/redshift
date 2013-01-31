# cython: profile=True
"""
Handle parser features
"""
from libc.stdlib cimport malloc, free
from libc.stdint cimport uint32_t

from io_parse cimport Sentence
from index.hashes cimport encode_feat

from _state cimport State, get_left_edge, get_right_edge

DEF CONTEXT_SIZE = 66

# There must be a way to keep this in synch??
N_LABELS = 0

cdef set_n_labels(int n):
    global N_LABELS
    N_LABELS = n

# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum

PAD_SIZE = 4

cdef enum:
    N0w
    N0p
    N0l
    N0lw
    N0lp
    N0ll
    N0lv
    N0l2w
    N0l2p
    N0l2l
    N1w
    N1p
    N1l
    N2w
    N2p
    N2l
    S0w
    S0p
    S0l
    S0hw
    S0hp
    S0hl
    S0lw
    S0lp
    S0ll
    S0rw
    S0rp
    S0rl
    S0rew
    S0rep
    S0rel
    S0l2w
    S0l2p
    S0l2l
    S0l2b
    S0r2w
    S0r2p
    S0r2l
    S0r2b
    S0h2w
    S0h2p
    S0h2l
    S0h2b
    S0lv
    S0rv
    dist
    S1w
    S1p
    S1l
    S1lw
    S1lp
    S1ll
    S1rew
    S1rep
    S1rel
    S1re_dist
    S_2w
    S_2p
    S_2l
    S_3w
    S_3p
    S_3l
    S0llabs
    S0rlabs
    N0llabs
    depth
    _context_size
assert CONTEXT_SIZE == _context_size, "Set CONTEXT_SIZE to %d in features.pyx" % _context_size

cdef int fill_context(size_t* context, size_t n0, size_t n1, n2,
                      size_t s0, size_t s1, size_t s_2, size_t s_3,
                      size_t s0_re, size_t s1_re,
                      size_t stack_len,
                      size_t* words, size_t* pos, size_t* browns,
                      size_t* heads, size_t* labels, size_t* l_vals, size_t* r_vals,
                      size_t* s0_lkids, size_t* s0_rkids, size_t* s1_lkids, size_t* s1_rkids,
                      size_t* n0_lkids,
                      bint* s0_llabels, bint* s0_rlabels, bint* n0_llabels) except -1:
    cdef size_t t, d

    context[N0w] = words[n0]
    context[N0p] = pos[n0]
    context[N0l] = labels[n0]

    context[N1w] = words[n1]
    context[N1p] = pos[n1]
    context[N1l] = labels[n1]

    context[N2w] = words[n2]
    context[N2p] = pos[n2]
    context[N2l] = labels[n2]

    context[S0w] = words[s0]
    context[S0p] = pos[s0]
    context[S0l] = labels[s0]

    context[S0hw] = words[heads[s0]]
    context[S0hp] = pos[heads[s0]]
    context[S0hl] = labels[heads[s0]]

    # The 'right edge' is the right-most branch of S0
    context[S0rew] = words[s0_re]
    context[S0rep] = pos[s0_re]
    context[S0rel] = labels[s0_re]

    context[S1w] = words[s1]
    context[S1p] = pos[s1]
    context[S1l] = labels[s1]
    
    # These are stack[2] and stack[3], not stack[-3] and stack[-4]!
    context[S_2w] = words[s_2]
    context[S_2p] = pos[s_2]
    context[S_2l] = labels[s_2]
    
    context[S_3w] = words[s_3]
    context[S_3p] = pos[s_3]
    context[S_3l] = labels[s_3]

    # Should this be leftmost??
    context[S1lw] = words[s1_lkids[0]]
    context[S1lp] = pos[s1_lkids[0]]
    context[S1ll] = labels[s1_lkids[0]]
    
    # The "right edge feature refers to the child before S0. If S0 is
    # attached to S1, then "right edge" is the second right-most child
    # of s.second. If S0 is not a child of S1, then it's the _rightmost_
    # child.
    # E.g. (S1 (S1re ) ) (S0) vs (S1 (S1re ) (S0))
    context[S1rew] = words[s1_re]
    context[S1rep] = pos[s1_re]
    context[S1rel] = labels[s1_re]
    # Token distance between S0 and its right neighbour
    if (s0 - s1_re) >= 5:
        context[S1re_dist] = 5
    else:
        context[S1re_dist] = s0 - s1_re

    context[S0lv] = l_vals[s0]
    context[S0rv] = r_vals[s0]
    context[N0lv] = l_vals[n0]
    
    t = s0_lkids[l_vals[s0] - 1]
    context[S0lw] = words[t]
    context[S0lp] = pos[t]
    context[S0ll] = labels[t]
    
    t = s0_rkids[r_vals[s0] - 1]
    context[S0rw] = words[t]
    context[S0rp] = pos[t]
    context[S0rl] = labels[t]
    
    if l_vals[s0] > 1:
        t = s0_lkids[l_vals[s0] - 2]
    else:
        t = 0
    context[S0l2w] = words[t]
    context[S0l2p] = pos[t]
    context[S0l2l] = labels[t]
    
    if r_vals[s0] > 1:
        t = s0_rkids[r_vals[s0] - 2]
    else:
        t = 0
    context[S0r2w] = words[t]
    context[S0r2p] = pos[t]
    context[S0r2l] = labels[t]

    if l_vals[n0] > 0:
        t = n0_lkids[l_vals[n0] - 1]
    else:
        t = 0
    context[N0lw] = words[t]
    context[N0lp] = pos[t]
    context[N0ll] = labels[t]
    
    if l_vals[n0] > 1:
        t = n0_lkids[l_vals[n0] - 2]
    else:
        t = 0
    context[N0l2w] = words[t]
    context[N0l2p] = pos[t]
    context[N0l2l] = labels[t]
    
    t = heads[heads[s0]]
    context[S0h2w] = words[t]
    context[S0h2p] = pos[t]
    context[S0h2l] = labels[t]
    
    context[S0llabs] = 0
    context[S0rlabs] = 0
    context[N0llabs] = 0
    for j in range(N_LABELS):
        context[S0llabs] += (s0_llabels[(N_LABELS - 1) - j] << j)
        context[S0rlabs] += (s0_rlabels[(N_LABELS - 1) - j] << j)
        context[N0llabs] += (n0_llabels[(N_LABELS - 1) - j] << j)

    d = n0 - s0
    # This makes sense as 0, not 1, because we do want to know how far away
    # we are from the OOB token. 
    if s0 == 0:
        context[dist] = 0
    elif d >= 5:
        context[dist] = 5
    else:
        context[dist] = d
    if stack_len >= 5:
        context[depth] = 5
    else:
        context[depth] = stack_len
    return 1


USE_LABELS_IN_EXTRA = False
def unigram(w, p, l):
    if USE_LABELS_IN_EXTRA:
        return ((w, p, l), (w, p), (w, l), (p, l), (w,), (p,), (l,))
    else:
        return ((w, p), (w,), (p,))

def bigram(w1, p1, l1, w2, p2, l2):
    if USE_LABELS_IN_EXTRA:
        raise StandardError
    else:
        return ((w1, w2), (w1, p2), (p1, w2), (p2, p2))

def brown_bigram(w1, p1, b1, b41, b61, w2, p2, b2, b42, b62):
    return (#(b41, b42),
            (b61, b62),
            (b1, b2),
            #(b41, p2),
            #(p1, b42),
            (b61, p2),
            (p1, b62),
            #(b41, w2),
            #(w1, b41),
            (b61, w2),
            (w1, b62),
            (b1, w2),
            (w1, b2))

def trigram(w1, p1, l1, w2, p2, l2, w3, p3, l3):
    if USE_LABELS_IN_EXTRA:
        raise StandardError
    else:
        return ((w1, w2, w3), (w1, w2, p3), (w1, p2, w3), (w1, p2, p3), (p1, w2, w3),
                (p1, w2, p3), (p1, p2, w3), (p1, p2, p3))

def brown_trigram(w1, p1, b1, b41, b61, w2, p2, b2, b42, b62, l3, p3, b3, b43, b63):
    return (
        (b41, b42, b43),
        (b61, b62, b63),
        (p1, b62, b63))


cdef Predicate* predicates
cdef int make_predicates(bint add_extra, bint add_labels) except 0:
    global N_PREDICATES, predicates
    cdef object feats, feat
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

    unigrams = (
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

    # Extra
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
        # For right-raise (and others)
        (S1p, S0p, N0p),
        (S1w, S0w, N0w),
        (S1w, S0p, N0p),
        (depth, S1w, N1w),
        (dist, S1w, N1w),
        (dist, S1p, N0p, N1p),
        # For right/left unshift
        (S0hp, S0w, S0p, S1w, S1p, S1l),
        (S0hp, S0p, S1w),
        (S0hp, S0w, S1p),
        # For left-invert
        (S0ll, S0w, N0w),
        (S0ll, S0w, N0p),
        (S0ll, S0p, N0w),
        #(S0lw, N0w),
        #(S0lp, N0p),
        #(S0lp, S0p, N0p),
        # For right-lower
        (S1rep, S0w, N0w),
        (S1rew, S0w, N0p),
        (S1rew, N0w),
        (S1rew, S0w),
        (S1re_dist,),
        (S1re_dist, S0w),
        (S1rep, S0p),
        # For "low-edge"
        (S0rew, N0w),
        (S0rep, N0w),
        (S0rew, N0p),
        # Found by accident!
        (S0w, N0lv),
        (S0p, N0lv),
        # Features for stack[2] and stack[3] are apparently good!!
        #(S_2w, N0w),
        #(S_2w, N1w),
        #(S_2p, N0p, N1w),
        #(S_2p, N0w, N1w),
        #(S_2w, N0p, N1p),
        #(S_3w, N0w),
        #(S_3w, N1w),
        #(S_3p, S_2p, N0w),

    )

    feats = from_single + from_word_pairs + from_three_words + distance + valency + unigrams + third_order
    if add_labels:
        print "Using label features"
        feats += labels
        feats += label_sets
    if add_extra:
        print "Using stack-second features"
        #feats += new_extra
        feats += stack_second
    N_PREDICATES = len(feats)
    predicates = <Predicate*>malloc(N_PREDICATES * sizeof(Predicate))
    for i, feat in enumerate(feats):
        predicates[i] = make_predicate(i, feat)
    return N_PREDICATES


cdef Predicate make_predicate(int id, object args):
    cdef int element
    cdef Predicate pred = Predicate(id=id, n=len(args))
    pred.raws = <size_t*>malloc(len(args) * sizeof(size_t))
    pred.args = <int*>malloc(len(args) * sizeof(int))
    for i, element in enumerate(args):
        pred.args[i] = element
    # TODO: Add estimates for each feature type
    pred.expected_size = 1000
    return pred


cdef size_t* init_context():
    return <size_t*>malloc(CONTEXT_SIZE * sizeof(size_t))


cdef size_t* init_hashed_features():
    return <size_t*>malloc(N_PREDICATES * sizeof(size_t))

cdef int extract(size_t* context, size_t* hashed,
        Sentence* sent, State* s) except -1:
    cdef int i, j
    cdef size_t out
    cdef Predicate predicate
    global predicates
    cdef size_t s_2 = s.stack[2]
    cdef size_t s_3 = s.stack[3]
    cdef size_t s0_re = get_right_edge(s, s.top)
    cdef size_t s1_re = get_right_edge(s, s.second)
    fill_context(context, s.i, s.i + 1, s.i + 2,
                 s.top, s.second, s_2, s_3, s0_re, s1_re, s.stack_len,
                 sent.words, sent.pos, sent.browns,
                 s.heads, s.labels, s.l_valencies, s.r_valencies,
                 s.l_children[s.top], s.r_children[s.top],
                 s.l_children[s.second], s.r_children[s.second],
                 s.l_children[s.i],
                 s.llabel_set[s.top], s.rlabel_set[s.top], s.llabel_set[s.i])

    cdef bint seen_non_zero
    for i in range(N_PREDICATES):
        predicate = predicates[i]
        seen_non_zero = False
        for j in range(predicate.n):
            predicate.raws[j] = context[predicate.args[j]]
            if predicate.raws[j] != 0:
                seen_non_zero = True
        if seen_non_zero or predicate.n == 1:
            out = encode_feat(predicate.raws, predicate.n, i)
            hashed[i] = out
        else:
            hashed[i] = 0
