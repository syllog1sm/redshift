# cython: profile=True
from _state cimport *
from transitions cimport TransitionSystem

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair

from cython.operator cimport preincrement as inc
from cython.operator cimport dereference as deref


cdef class Beam:
    def __cinit__(self, TransitionSystem trans, 
                  size_t k, size_t length, upd_strat='early'):
        self.trans = trans
        self.upd_strat = upd_strat
        self.length = length
        self.k = k
        self.i = 0
        self.nr_skip = 0
        self.is_finished = False
        self.ancestry = <size_t*>calloc(k, sizeof(size_t))
        self.anc_freqs = <size_t**>malloc(k * sizeof(size_t))
        cdef size_t i
        for i in range(self.k):
            self.anc_freqs[i] = <size_t*>calloc(k, sizeof(size_t))
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        for i in range(k):
            self.parents[i] = init_state(length)
        for i in range(k):
            self.beam[i] = init_state(length)
        self.gold = init_state(length)
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.costs = <int**>malloc(self.k * sizeof(int*))
        self.valid = <bint**>malloc(self.k * sizeof(bint*))
        for i in range(self.k):
            self.costs[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))
            self.valid[i] = <bint*>calloc(self.trans.nr_class, sizeof(bint*))
        self.violn = None

    cdef Kernel* next_state(self, size_t idx):
        self.trans.fill_valid(self.beam[idx], self.valid[idx])
        fill_kernel(self.beam[idx])
        return &self.beam[idx].kernel

    cdef int cost_next(self, size_t i, size_t* heads, size_t* labels) except -1:
        self.trans.fill_static_costs(self.beam[i], heads, labels, self.costs[i])
        fill_kernel(self.beam[i])

    cdef int extend_states(self, double** ext_scores) except -1:
        global merged
        # Former states are now parents, beam will hold the extensions
        cdef State** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        self.psize = self.bsize
        self.bsize = 0
        cdef size_t parent_idx, clas, move_id
        cdef double parent_score, score
        cdef double* scores
        cdef priority_queue[pair[double, size_t]] next_moves = priority_queue[pair[double, size_t]]()
        # Get best parent/clas pairs by score
        for parent_idx in range(self.psize):
            parent_score = self.parents[parent_idx].score
            scores = ext_scores[parent_idx]
            for clas in range(self.trans.nr_class):
                if self.valid[parent_idx][clas] != -1:
                    score = parent_score + scores[clas]
                    move_id = (parent_idx * self.trans.nr_class) + clas
                    next_moves.push(pair[double, size_t](score, move_id))
        cdef pair[double, size_t] data
        # Apply extensions for best continuations
        cdef State* s
        cdef State* parent
        cdef uint64_t key
        cdef dense_hash_map[uint64_t, int] seen_states = dense_hash_map[uint64_t, int](self.k)
        seen_states.set_empty_key(0)
        while self.bsize < self.k and not next_moves.empty():
            data = next_moves.top()
            parent_idx = data.second / self.trans.nr_class
            assert parent_idx < self.psize
            clas = data.second % self.trans.nr_class
            parent = self.parents[parent_idx]
            # We've got two arrays of states, and we swap beam-for-parents.
            # So, s here will get manipulated, then copied into parents later.
            s = self.beam[self.bsize]
            copy_state(s, parent)
            s.cost += self.costs[parent_idx][clas]
            s.score = data.first
            self.trans.transition(clas, s)
            self.anc_freqs[self.ancestry[parent_idx]][parent_idx] += 1
            self.ancestry[self.bsize] = parent_idx
            # Unless! If s has an identical "signature" to a previous state,
            # then we know it's dominated, and we can discard it. We do that by
            # just not advancing self.bsize, as that means this s struct
            # will be reused next iteration, and over-written.
            key = MurmurHash64A(s.sig, (s.i + 1) * sizeof(size_t), 0)
            if seen_states[key] == 0:
                seen_states[key] = 1
                self.bsize += 1
            else:
                self.nr_skip += 1
            next_moves.pop()
        self.is_full = self.bsize >= self.k
        # Flush next_moves queue
        self.t += 1
        if self.beam[0].is_finished:
            self.is_finished = True

    cdef bint check_violation(self):
        cdef Violation violn
        cdef bint out_of_beam
        if self.bsize < self.k:
            return False
        if self.beam[0].cost == 0:
            return False
        if self.gold.score > self.beam[0].score:
            return False
        if self.upd_strat == 'early' and self.violn != None:
            return False
        out_of_beam = True
        for i in range(self.bsize):
            if self.beam[i].cost == 0:
                out_of_beam = False
                gold = self.beam[i]
                break
        else:
            gold = self.gold
        violn = Violation()
        violn.set(self.beam[0], gold, out_of_beam)
        if self.upd_strat == 'max':
            if self.violn is None or violn.delta > self.violn.delta:
                self.violn = violn
        if out_of_beam and self.upd_strat == 'early' and self.violn == None:
            self.violn = violn
        return self.upd_strat == 'early' and bool(self.violn)

    cdef int fill_parse(self, size_t* hist, size_t* heads, size_t* labels) except -1:
        for i in range(self.t):
            hist[i] = self.beam[0].history[i]
        # No need to copy heads for root and start symbols
        for i in range(1, self.length - 1):
            assert self.beam[0].heads[i] != 0
            heads[i] = self.beam[0].heads[i]
            labels[i] = self.beam[0].labels[i]

    def __dealloc__(self):
        free_state(self.gold)
        for i in range(self.k):
            free_state(self.beam[i])
            free_state(self.parents[i])
            free(self.valid[i])
            free(self.costs[i])
            free(self.anc_freqs[i])
        free(self.anc_freqs)
        free(self.ancestry)
        free(self.beam)
        free(self.parents)
        free(self.valid)
        free(self.costs)

"""
cdef class FastBeam(Beam):
    cdef FastState** parents
    cdef FastState** beam
    cdef priority_queue[pair[double, size_t]]* next_moves
    cdef State* gold
    cdef object upd_strat
    cdef size_t n_labels
    cdef size_t max_class
    cdef size_t nr_class
    cdef size_t k
    cdef size_t i
    cdef size_t bsize
    cdef size_t psize
    cdef Violation first_violn
    cdef Violation max_violn
    cdef Violation last_violn
    cdef Violation cost_violn
    cdef bint is_full
    cdef bint add_labels
    cdef Cont** conts
    cdef bint** seen_moves

    cdef FastState* add(self, size_t par_idx, double score, int cost) except NULL:
        cdef FastState* parent = self.parents[par_idx]
        parent.nr_kids += 1
        assert par_idx < self.psize
        assert not self.is_full

        #copy_state(self.beam[self.bsize], parent)
        cdef FastState* ext = <FastState*>malloc(sizeof(FastState))
        self.beam[self.bsize] = ext
        ext.last_action = cont.clas
        ext.previous = parent
        if cont.clas == SHIFT:
            ext.k = kernel_from_s(parent.k)
            ext.tail = parent
        elif cont.clas == RIGHT:
            ext.k = kernel_from_r(parent.k, cont.label)
            ext.tail = parent
        elif cont.clas == REDUCE:
            ext.k = kernel_from_d(parent, parent.tail)
            ext.tail = parent.tail
        elif cont.clas == LEFT:
            ext.k = kernel_from_l(parent, parent.tail, cont.label)
            ext.previous = parent
            ext.tail = parent.tail
        ext.score = score
        ext.is_gold = parent.is_gold and cost == 0
        ext.cost += cost
        self.bsize += 1
        self.is_full = self.bsize >= self.k
        return ext

    cdef refresh(self):
        cdef size_t i, j
        for i in range(self.max_class):
            for j in range(N_MOVES):
                self.seen_moves[i][j] = False
        for i in range(self.bsize):
            if self.beam[i].nr_kids == 0:
                free_fast_state(self.beam[i])
        cdef State** parents = self.parents
        self.parents = self.beam
        self.beam = parents
        del self.next_moves
        self.next_moves = new priority_queue[pair[double, size_t]]()
        self.psize = self.bsize
        self.is_full = False
        self.bsize = 0
        self.i = 0

    def __dealloc__(self):
        for i in range(self.bsize):
            free_fast_state(self.beam[i])
        for i in range(self.psize):
            free_fast_state(self.parents[i])
        for i in range(self.max_class):
            free(self.seen_moves[i])
            free(self.conts[i])
        free(self.beam)
        free(self.parents)
        free(self.conts)
        free(self.seen_moves)
        free_state(self.gold)
"""


cdef class Violation:
    """
    A gold/prediction pair where the g.score < p.score
    """

    def __cinit__(self):
        self.out_of_beam = False
        self.t = 0
        self.delta = 0.0
        self.cost = 0

    cdef int set(self, State* p, State* g, bint out_of_beam) except -1:
        self.delta = p.score - g.score
        self.cost = p.cost
        assert g.t == p.t, '%d vs %d' % (g.t, p.t)
        self.t = g.t
        self.ghist = <size_t*>malloc(self.t * sizeof(size_t))
        memcpy(self.ghist, g.history, self.t * sizeof(size_t))
        self.phist = <size_t*>malloc(self.t * sizeof(size_t))
        memcpy(self.phist, p.history, self.t * sizeof(size_t))
        self.out_of_beam = out_of_beam

    def __dealloc__(self):
        free(self.ghist)
        free(self.phist)


