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
                  size_t k, size_t length, upd_strat='early', prune_freqs=None):
        self.trans = trans
        self.upd_strat = upd_strat
        self.length = length
        self.k = k
        self.i = 0
        self.nr_skip = 0
        self.is_finished = False
        cdef size_t i
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
        self.valid = <int**>malloc(self.k * sizeof(bint*))
        for i in range(self.k):
            self.costs[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))
            self.valid[i] = <int*>calloc(self.trans.nr_class, sizeof(int*))
        self.violn = None
        self._prune_freqs = prune_freqs

    cdef Kernel* gold_kernel(self, size_t* tags):
        fill_kernel(self.gold, tags)
        return &self.gold.kernel

    cdef int advance_gold(self, double* scores, size_t* tags,
                          size_t* heads, size_t* labels) except -1:
        cdef size_t oracle
        cdef double best_score = -100000
        cdef int* costs = self.trans.get_costs(self.gold, tags, heads, labels)
        cdef bint use_dyn_amb = True
        if use_dyn_amb:
            for i in range(self.trans.nr_class):
                if scores[i] >= best_score and costs[i] == 0:
                    oracle = i
                    best_score = scores[i]
        else:
            oracle = self.trans.break_tie(self.gold, tags, heads, labels)

        self.gold.score += scores[oracle]
        self.trans.transition(oracle, self.gold)

    cdef Kernel* next_state(self, size_t idx, size_t* tags):
        self.trans.fill_valid(self.beam[idx], self.valid[idx])
        fill_kernel(self.beam[idx], tags)
        return &self.beam[idx].kernel

    cdef int cost_next(self, size_t i, size_t* tags, size_t* heads, size_t* labels) except -1:
        #self.trans.fill_static_costs(self.beam[i], tags, heads, labels, self.costs[i])
        cdef int* costs = self.trans.get_costs(self.beam[i], tags, heads, labels)
        memcpy(self.costs[i], costs, sizeof(int) * self.trans.nr_class)
        fill_kernel(self.beam[i], tags)

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
            self.bsize += 1
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
        if gold.score > self.beam[0].score:
            return None
        violn = Violation()
        violn.set(self.beam[0], gold, out_of_beam)
        if self.upd_strat == 'max':
            if self.violn is None or violn.delta > self.violn.delta:
                self.violn = violn
        if out_of_beam and self.upd_strat == 'early' and self.violn == None:
            self.violn = violn
        return self.upd_strat == 'early' and bool(self.violn)

    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd) except -1:
        cdef size_t rightmost = 1
        # No need to copy heads for root and start symbols
        for i in range(1, self.length - 1):
            assert self.beam[0].heads[i] != 0
            #tags[i] = self.beam[0].tags[i]
            heads[i] = self.beam[0].heads[i]
            labels[i] = self.beam[0].labels[i]
            # Do sentence boundary detection
            # TODO: Set this as ROOT label
            #raise StandardError
        survivors = set()
        cdef State* s
        if self._prune_freqs is not None:
            for idx in range(self.bsize):
                prefix = []
                s = self.beam[idx]
                for i in range(s.t):
                    prefix.append(s.history[i])
                    key = tuple(prefix)
                    if key not in survivors:
                        survivors.add(key)
                        for survived_for in range(1, s.t - len(key)):
                            self._prune_freqs.setdefault(survived_for, 0)
                            self._prune_freqs[survived_for] += 1
 
    def __dealloc__(self):
        free_state(self.gold)
        for i in range(self.k):
            free_state(self.beam[i])
            free_state(self.parents[i])
            free(self.valid[i])
            free(self.costs[i])
        free(self.beam)
        free(self.parents)
        free(self.valid)
        free(self.costs)

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


