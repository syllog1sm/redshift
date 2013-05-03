from _state cimport *
from transitions cimport TransitionSystem

from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport uint64_t, int64_t

from libcpp.utility cimport pair
from libcpp.vector cimport vector
from libcpp.queue cimport priority_queue



cdef class Beam:
    def __cinit__(self, TransitionSystem trans, 
                  size_t k, size_t length, upd_strat='early'):
        cdef size_t i
        cdef Cont* cont
        cdef State* s
        self.trans = trans
        self.upd_strat = upd_strat
        self.k = k
        self.i = 0
        self.parents = <State**>malloc(k * sizeof(State*))
        self.beam = <State**>malloc(k * sizeof(State*))
        for i in range(k):
            self.parents[i] = init_state(length)
        for i in range(k):
            self.beam[i] = init_state(length)
        self.gold = init_state(length)
        self.bsize = 1
        self.psize = 0
        self.is_full = self.bsize >= self.k
        self.max_class = self.trans.nr_class * k
        self.next_moves = new priority_queue[pair[double, size_t]]()
        self.conts = <Cont**>malloc(self.max_class * sizeof(Cont*))
        for i in range(self.max_class):
            self.conts[i] = <Cont*>malloc(sizeof(Cont))
        self.violn = None

    cdef int add(self, size_t par_idx, double score, int cost,
                 size_t clas, size_t rlabel) except -1:
        cdef State* parent = self.parents[par_idx]
        assert par_idx < self.psize
        assert not self.is_full
        copy_state(self.beam[self.bsize], parent)
        cdef State* ext = self.beam[self.bsize]
        ext.score = score
        ext.is_gold = ext.is_gold and cost == 0
        ext.cost += cost
        self.bsize += 1
        self.is_full = self.bsize >= self.k
        ext.guess_labels[ext.i] = rlabel
        self.trans.transition(clas, ext)
        fill_kernel(ext)

    cdef int extend(self, size_t parent_idx, double* scores) except -1:
        cdef double best_right_score = scores[self.trans.r_start]
        cdef size_t best_right = self.trans.labels[self.trans.r_start]
        cdef size_t i
        for i in range(self.trans.r_start + 1, self.trans.r_end):
            if scores[i] > best_right_score:
                best_right_score = scores[i]
                best_right = self.trans.labels[i]
        cdef State* parent = self.parents[parent_idx]
        cdef double parent_score = parent.score
        cdef Cont* cont
        cdef size_t clas
        for clas in range(self.trans.nr_class):
            if not self.trans.is_valid(clas, parent.i, parent.n, parent.stack_len,
                                       parent.heads[parent.top]):
                continue
            cont = self.conts[self.i]
            self.i += 1
            cont.score = parent_score + scores[clas]
            cont.parent = parent_idx
            cont.clas = clas
            cont.rlabel = best_right
            self.next_moves.push(pair[double, size_t](cont.score, <size_t>cont))

    cdef bint check_violation(self):
        cdef Violation violn
        cdef bint out_of_beam
        if self.bsize < self.k:
            return False
        if self.beam[0].is_gold:
            return False
        if self.gold.score > self.beam[0].score:
            return False
        if self.upd_strat == 'early' and self.violn != None:
            return False
        out_of_beam = True
        for i in range(self.bsize):
            if self.beam[i].is_gold:
                out_of_beam = False
                gold = self.beam[i]
                break
        else:
            gold = self.gold
        violn = Violation()
        violn.set(self.beam[0], gold, out_of_beam)
        if self.upd_strat == 'max' and violn.delta > self.max_violn.delta:
            self.violn = violn
        if out_of_beam and self.upd_strat == 'early' and self.violn == None:
            self.violn = violn
        return self.upd_strat == 'early' and bool(self.violn)

    cdef State* best_p(self) except NULL:
        if self.bsize != 0:
            return self.beam[0]
        else:
            raise StandardError

    cdef refresh(self):
        cdef size_t i, j
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
            free_state(self.beam[i])
        for i in range(self.psize):
            free_state(self.parents[i])
        for i in range(self.max_class):
            free(self.conts[i])
        free(self.beam)
        free(self.parents)
        free(self.conts)
        free_state(self.gold)

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


