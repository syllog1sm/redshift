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

# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum

cdef enum:
    N0w
    N0p
    N0lw
    N0lp
    N0ll
    N0lv
    N0l2w
    N0l2p
    N0l2l
    N1w
    N1p
    N2w
    N2p
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
    S0lv
    S0rv
    S1w
    S1p
    dist
    S0llabs
    S0rlabs
    N0llabs
    CONTEXT_SIZE


cdef void fill_context(size_t* context, size_t nr_label, size_t* words, size_t* pos,
                       Kernel* k, Subtree* s0l, Subtree* s0r, Subtree* n0l):
    context[N0w] = words[k.i]
    context[N0p] = pos[k.i]

    context[N1w] = words[k.i + 1]
    context[N1p] = pos[k.i + 1]

    context[N2w] = words[k.i + 2]
    context[N2p] = pos[k.i + 2]

    context[S0w] = words[k.s0]
    context[S0p] = pos[k.s0]
    context[S0l] = k.Ls0

    context[S1w] = words[k.s1]
    context[S1p] = words[k.s1]
    
    context[S0hw] = words[k.hs0]
    context[S0hp] = pos[k.hs0]
    context[S0hl] = k.Lhs0

    context[S0h2w] = words[k.h2s0]
    context[S0h2p] = pos[k.h2s0]
    context[S0h2l] = k.Lh2s0
 
    context[S0lv] = s0l.val
    context[S0rv] = s0r.val
    context[N0lv] = n0l.val

    context[S0lw] = words[s0l.idx[0]]
    context[S0lp] = pos[s0l.idx[0]]
    context[S0rw] = words[s0r.idx[0]]
    context[S0rp] = pos[s0r.idx[0]]

    context[S0l2w] = words[s0l.idx[1]]
    context[S0l2p] = pos[s0l.idx[1]]
    context[S0r2w] = words[s0r.idx[1]]
    context[S0r2p] = pos[s0r.idx[1]]

    context[N0lw] = words[n0l.idx[0]]
    context[N0lp] = pos[n0l.idx[0]]
    context[N0l2w] = words[n0l.idx[1]]
    context[N0l2p] = pos[n0l.idx[1]]

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


cdef class FeatureSet:
    def __cinit__(self, nr_label, bint add_extra=False):
        self.nr_label = nr_label
        self._make_predicates(add_extra)
        self.context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        self.features = <uint64_t*>calloc(self.n, sizeof(uint64_t))
        self.min_feats = False

    def __dealloc__(self):
        free(self.context)
        free(self.features)
        free(self.predicates)

    cdef uint64_t* extract(self, Sentence* sent, Kernel* k) except NULL:
        cdef size_t* context = self.context
        assert <size_t>k != 0
        fill_context(context, self.nr_label, sent.words, sent.pos, k,
                     &k.s0l, &k.s0r, &k.n0l)
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
        features[f] = 0
        return features

    def _make_predicates(self, bint add_extra):
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

        unigrams = (
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
            (w, S1w, S0l),
            (w, S1p, S0l),
            (ww, S1w, S0w, S0l),
            (pp, S1p, S0p, S0l),
            (wp, S1w, S0p, S0l),
            (wp, S1p, S0w, S0l),
        )

        feats = from_single + from_word_pairs + from_three_words + distance + valency + unigrams + third_order
        feats += labels
        feats += label_sets
        if add_extra:
            print "Extra feats"
            feats += extra
        assert len(set(feats)) == len(feats), '%d vs %d' % (len(set(feats)), len(feats))
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
