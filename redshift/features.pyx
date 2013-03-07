"""
Handle parser features
"""
from libc.stdlib cimport malloc, free, calloc
from libc.stdint cimport uint64_t
from libcpp.pair cimport pair
import index.hashes
from cython.operator cimport dereference as deref, preincrement as inc


from io_parse cimport Sentence
#from index.hashes cimport encode_feat

from libcpp.vector cimport vector

DEF CONTEXT_SIZE = 53

# Context elements
# Ensure _context_size is always last; it ensures our compile-time setting
# is in synch with the enum

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
    S0llabs
    S0rlabs
    N0llabs
    depth
    _context_size
assert CONTEXT_SIZE == _context_size, "Set CONTEXT_SIZE to %d in features.pyx" % _context_size

cdef void fill_context(size_t* context, size_t nr_label, size_t n0, size_t n1, size_t n2,
                      size_t s0, size_t s1, size_t stack_len,
                      size_t* words, size_t* pos, size_t* browns,
                      size_t* heads, size_t* labels, size_t* l_vals, size_t* r_vals,
                      size_t* s0_lkids, size_t* s0_rkids, size_t* s1_lkids, size_t* s1_rkids,
                      size_t* n0_lkids,
                      bint* s0_llabels, bint* s0_rlabels, bint* n0_llabels):
    cdef uint64_t t, d, j

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

    context[S1w] = words[s1]
    context[S1p] = pos[s1]
    context[S1l] = labels[s1]
    
    # Should this be leftmost??
    context[S1lw] = words[s1_lkids[0]]
    context[S1lp] = pos[s1_lkids[0]]
    context[S1ll] = labels[s1_lkids[0]]
    context[S0lv] = l_vals[s0]
    context[S0rv] = r_vals[s0]
    context[N0lv] = l_vals[n0]
    t = s0_lkids[l_vals[s0] - 1] if l_vals[s0] > 0 else 0
    context[S0lw] = words[t]
    context[S0lp] = pos[t]
    context[S0ll] = labels[t]
    t = s0_rkids[r_vals[s0] - 1] if r_vals[s0] > 0 else 0
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
    for j in range(nr_label):
        # Decode the binary arrays representing the label sets into integers
        # Iterate in reverse, incrementing by the bit shifted by the idx
        context[S0llabs] += (s0_llabels[(nr_label - 1) - j] << j)
        context[S0rlabs] += (s0_rlabels[(nr_label - 1) - j] << j)
        context[N0llabs] += (n0_llabels[(nr_label - 1) - j] << j)
    d = n0 - s0
    # TODO: Seems hard to believe we want to keep d non-zero when there's no
    # stack top. Experiment with this futrther.
    if s0 != 0:
        context[dist] = d
    else:
        context[dist] = 0
    if stack_len >= 5:
        context[depth] = 5
    else:
        context[depth] = stack_len


cdef class FeatureSet:
    def __cinit__(self, nr_label, bint add_extra=False):
        self.nr_label = nr_label
        # Sets predicates, n, nr_multi, nr_uni. Allocates unigram arrays
        self._make_predicates(add_extra)
        self.context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        self.features = <uint64_t*>calloc(self.n, sizeof(uint64_t))
        cdef dense_hash_map[uint64_t, uint64_t] *table
        self.i = 1
        self.save_entries = False
        self.unigrams = vector[dense_hash_map[uint64_t, uint64_t]]()
        for i in range(self.nr_uni):
            table = new dense_hash_map[uint64_t, uint64_t]()
            self.unigrams.push_back(table[0])
            self.unigrams[i].set_empty_key(0)
        self.tables = vector[dense_hash_map[uint64_t, uint64_t]]()
        for i in range(self.nr_multi):
            table = new dense_hash_map[uint64_t, uint64_t]()
            self.tables.push_back(table[0])
            self.tables[i].set_empty_key(0)

    def __dealloc__(self):
        free(self.context)
        free(self.features)
        free(self.predicates)

    cdef uint64_t* extract(self, Sentence* sent, State* s):
        cdef size_t* context = self.context
        fill_context(context, self.nr_label, s.i, s.i + 1, s.i + 2,
                     s.top, s.second, s.stack_len,
                     sent.words, sent.pos, sent.browns,
                     s.heads, s.labels, s.l_valencies, s.r_valencies,
                     s.l_children[s.top], s.r_children[s.top],
                     s.l_children[s.second], s.r_children[s.second],
                     s.l_children[s.i], s.llabel_set[s.top], s.rlabel_set[s.top],
                     s.llabel_set[s.i])
        cdef size_t f = 0
        cdef size_t i
        cdef uint64_t hashed
        cdef uint64_t feat
        cdef uint64_t value
        cdef uint64_t* features = self.features
        for i in range(self.nr_uni):
            value = context[self.uni_feats[i]]
            if value == 0:
                continue
            feat = self.unigrams[i][value]
            if feat != 0:
                features[f] = feat
                f += 1
            elif self.save_entries:
                self.unigrams[i][value] = self.i
                features[f] = self.i
                f += 1
                self.i += 1

        cdef uint64_t j
        cdef bint seen_non_zero
        cdef Predicate* pred
        cdef size_t n
        for i in range(self.nr_multi):
            pred = &self.predicates[i]
            seen_non_zero = False
            for j in range(pred.n):
                value = context[pred.args[j]]
                pred.raws[j] = value
                if value != 0:
                    seen_non_zero = True
            if seen_non_zero:
                hashed = MurmurHash64A(pred.raws, <uint64_t>pred.n * sizeof(uint64_t), i)
                feat = self.tables[i][hashed]
                if feat != 0:
                    features[f] = feat
                    f += 1
                elif self.save_entries:
                    self.tables[i][hashed] = self.i
                    features[f] = self.i
                    self.i += 1
                    f += 1
        features[f] = 0
        return features
   
    def save(self, path):
        cdef pair[uint64_t, uint64_t] data
        cdef dense_hash_map[uint64_t, uint64_t].iterator it
        out = open(str(path), 'w')
        for i in range(self.nr_uni):
            it = self.unigrams[i].begin()
            while it != self.unigrams[i].end():
                data = deref(it)
                out.write('u\t%d,%d\t%d\n' % (i, data.first, data.second))
                inc(it)
        for i in range(self.nr_multi):
            it = self.tables[i].begin()
            while it != self.tables[i].end():
                data = deref(it)
                out.write('%d\t%d\t%d\n' % (i, data.first, data.second))
                inc(it)
        out.close()
                
    def load(self, path):
        cdef uint64_t value
        for line in open(str(path)):
            fields = line.strip().split()
            i = fields[0]
            hashed = fields[1]
            value = int(fields[2])
            if i == 'u':
                i, hashed = hashed.split(',')
                self.unigrams[int(i)][int(hashed)] = <size_t>value
            else:
                self.tables[int(i)][int(hashed)] = value


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

        # Extra
        stack_second = (
            # For reattach. We need these because if we left-clobber, we need to
            # know what will be our head
            (w, S1w,),
            (p, S1p,),
            (wp, S1w, S1p),
            (ww, S1w, N0w),
            (wp, S1w, N0p),
            (wp, S1p, N0w),
            (pp, S1p, N0p),
            (ww, S1w, N1w),
            (wp, S1w, N1p),
            (pp, S1p, N1p),
            (pw, S1p, N1w),
            (dww, dist, S1w, N1w),
            (dppp, dist, S1p, N0p, N1p),
            # For right-raise (and others)
            #(S1p, S0p, N0p),
            #(S1w, S0w, N0w),
            #(S1w, S0p, N0p),
            #(depth, S1w, N1w),
            # For right/left unshift
            #(S0hp, S0w, S0p, S1w, S1p, S1l),
            (wpp, S0hp, S0p, S1w),
            (wpp, S0hp, S0w, S1p),
            # For left-invert
            (lww, S0ll, S0w, N0w),
            (lww, S0ll, S0w, N0p),
            (lwp, S0ll, S0p, N0w),
            (ww, S0lw, N0w),
            (pp, S0lp, N0p),
            (ppp, S0lp, S0p, N0p),
            (vw, S0w, N0lv),
            (vp, S0p, N0lv),
            # For right-lower
            #(S1rep, S0w, N0w),
            #(S1rew, S0w, N0p),
            #(S1rew, N0w),
            #(S1rew, S0w),
            #(S1re_dist,),
            #(S1re_dist, S0w),
            #(S1rep, S0p),
            # For "low-edge"
            #(S0rew, N0w),
            #(S0rep, N0w),
            #(S0rew, N0p),
            # Found by accident
            # For new right lower
            (ww, S0r2w, S0rw),
            (pp, S0r2p, S0rp),
            (wp, S0r2w, S0rp),
            (wp, S0r2p, S0rw),
            (ww, S0w, S0rw),
            (pw, S0w, S0rp),
            (pp, S0p, S0rp),
            (pp, S0p, S0rw),
            (pp, S0p, S0rp),
            (wwp, S0p, S0r2w, S0rw),
            (ppp, S0p, S0r2p, S0rp),
            (wpp, S0p, S0rp, N0w),
            (ppp, S0p, S0rp, N0p),
            (wpp, S0w, S0rp, N0p),
        )

        feats = from_single + from_word_pairs + from_three_words + distance + valency + unigrams + third_order
        feats += labels
        feats += label_sets
        if add_extra:
            print "Using stack-second features"
            feats += stack_second
        assert len(set(feats)) == len(feats)
        self.n = len(feats)
        uni_feats = list(sorted([f for f in feats if len(f) == 2], key=lambda i: i[1]))
        multi_feats = list(sorted([f for f in feats if len(f) > 2], key=lambda i: i[1]))
        self.nr_uni = len(uni_feats)
        #self.unigrams = <uint64_t**>malloc(self.nr_uni * sizeof(uint64_t*))
        self.uni_feats = <size_t*>malloc(self.nr_uni * sizeof(size_t))
        #self.uni_lens = <size_t*>malloc(self.nr_uni * sizeof(size_t))
        for i, feat in enumerate(uni_feats):
        #    self.unigrams[i] = <uint64_t*>calloc(feat[0], sizeof(uint64_t))
        #    self.uni_lens[i] = feat[0]
            self.uni_feats[i] = feat[1]
        self.nr_multi = len(multi_feats)
        self.predicates = <Predicate*>malloc(self.nr_multi * sizeof(Predicate))
        cdef Predicate pred
        for id_, args in enumerate(multi_feats):
            size = args[0]
            args = args[1:]
            pred = Predicate(id=id_, n=len(args), expected_size = size)
            pred.raws = <uint64_t*>malloc(len(args) * sizeof(uint64_t))
            pred.args = <int*>malloc(len(args) * sizeof(int))
            for i, element in enumerate(sorted(args)):
                pred.args[i] = element
            self.predicates[id_] = pred

