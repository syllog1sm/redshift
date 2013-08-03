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

from ext.murmurhash cimport MurmurHash64A
from ext.sparsehash cimport *


cdef class Beam:
    cdef int swap_beam(self):
        cdef void** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        self.psize = self.bsize
        self.bsize = 0 

    cdef int extend_states(self, double** ext_scores) except -1:
        # Former states are now parents, beam will hold the extensions
        self.swap_beam()
        cdef size_t parent_idx, clas, move_id
        cdef double parent_score, score
        cdef double* scores
        cdef priority_queue[pair[double, size_t]] next_moves
        next_moves = priority_queue[pair[double, size_t]]()
        for parent_idx in range(self.psize):
            parent_score = self.get_score(parent_idx)
            # Account for variable-length transition histories
            if self._is_finished(0, parent_idx):
                move_id = (parent_idx * self.nr_class) + 0
                next_moves.push(pair[double, size_t](parent_score, move_id))
                continue
            scores = ext_scores[parent_idx]
            for clas in range(self.nr_class):
                if self.valid[parent_idx][clas] != -1:
                    score = parent_score + scores[clas]
                    move_id = (parent_idx * self.nr_class) + clas
                    next_moves.push(pair[double, size_t](score, move_id))
        cdef pair[double, size_t] data
        # Apply extensions for best continuations
        cdef uint64_t hashed = 0
        cdef dense_hash_map[uint64_t, bint] seen_equivs = dense_hash_map[uint64_t, bint]()
        seen_equivs.set_empty_key(0)
        while self.bsize < self.k and not next_moves.empty():
            data = next_moves.top()
            parent_idx = data.second / self.nr_class
            assert parent_idx < self.psize
            clas = data.second % self.nr_class
            hashed = self.extend_state(parent_idx, self.bsize, clas, data.first)
            # Ignore dominated extensions --- this is an alternative to having an
            # equivalence class; we simply don't build the other members of the class
            if hashed == 0 or not seen_equivs[hashed]:
                self.bsize += 1
                seen_equivs[hashed] = 1
            next_moves.pop()
        self.is_full = self.bsize >= self.k
        self.t += 1
        for i in range(self.bsize):
            if not self._is_finished(1, i):
                self.is_finished = False
                break
        else:
            self.is_finished = True

    cdef bint _is_finished(self, int p_or_b, size_t idx):
        raise NotImplementedError

    cdef uint64_t extend_state(self, size_t parent_idx, size_t b_idx,
            size_t clas, double score):
        raise NotImplementedError

    cdef int init_beams(self, size_t k, size_t length) except -1:
        raise NotImplementedError

    cdef double get_score(self, size_t parent_idx):
        raise NotImplementedError

    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd, bint* edits) except -1:
        raise NotImplementedError
 

cdef class ParseBeam(Beam):
    def __cinit__(self, object trans, 
                  size_t k, size_t length, nr_tag=None):
        self.trans = trans
        self.length = length
        self.k = k
        self.i = 0
        self.is_finished = False
        cdef size_t i
        self.parents = <void**>malloc(k * sizeof(void*))
        self.beam = <void**>malloc(k * sizeof(void*))
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.valid = <int**>malloc(self.k * sizeof(int*))
        self.costs = <int**>malloc(self.k * sizeof(int*))
        for i in range(self.k):
            self.valid[i] = <int*>calloc(self.nr_class, sizeof(int))
            self.costs[i] = <int*>calloc(self.nr_class, sizeof(int))
        self.init_beams(k, self.length)

    cdef int init_beams(self, size_t k, size_t length) except -1:
        cdef State* s
        for i in range(k):
            s = init_state(length)
            self.parents[i] = <void*>s
        for i in range(k):
            s = init_state(length)
            self.beam[i] = <void*>s

    cdef bint _is_finished(self, int p_or_b, size_t idx):
        cdef State* s
        if p_or_b == 0:
            s = <State*>self.parents[idx]
        else:
            s = <State*>self.beam[idx]
        return s.is_finished

    cdef double get_score(self, size_t parent_idx):
        cdef State* s = <State*>self.parents[parent_idx]
        # Account for variable-length transition histories
        if s.is_finished:
            mean_score = s.score / s.t
            return s.score + mean_score
        else:
            return s.score

    cdef uint64_t extend_state(self, size_t parent_idx, size_t b_idx, size_t clas,
                         double score):
        parent = <State*>self.parents[parent_idx]
        # We've got two arrays of states, and we swap beam-for-parents.
        # So, s here will get manipulated, then copied into parents later.
        s = <State*>self.beam[self.bsize]
        copy_state(s, parent)
        s.score = score
        if not s.is_finished:
            assert self.costs[parent_idx][clas] != -1
            s.cost += self.costs[parent_idx][clas]
            self.trans.transition(clas, s)
        return 0

    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd, bint* edits) except -1:
        cdef size_t i
        cdef State* s = <State*>self.beam[0]
        # No need to copy heads for root and start symbols
        for i in range(1, self.length - 1):
            assert s.heads[i] != 0
            #tags[i] = self.beam[0].tags[i]
            heads[i] = s.heads[i]
            labels[i] = s.labels[i]
            # TODO: Do sentence boundary detection here
        fill_edits(s, edits)

    def __dealloc__(self):
        for i in range(self.k):
            free_state(<State*>self.beam[i])
            free_state(<State*>self.parents[i])
            free(self.valid[i])
            free(self.costs[i])
        free(<State*>self.beam)
        free(<State*>self.parents)
        free(self.valid)
        free(self.costs)


cdef class TaggerBeam(Beam):
    def __cinit__(self, _, size_t k, size_t length, nr_tag=None):
        self.trans = None
        self.nr_class = nr_tag
        self.length = length
        self.k = k
        self.i = 0
        self.is_finished = False
        cdef size_t i
        self.bsize = 1
        self.psize = 0
        self.t = 0
        self.is_full = self.bsize >= self.k
        self.valid = <int**>malloc(self.k * sizeof(int*))
        self.costs = <int**>malloc(self.k * sizeof(int*))
        for i in range(self.k):
            self.valid[i] = <int*>calloc(self.nr_class, sizeof(int))
            self.costs[i] = <int*>calloc(self.nr_class, sizeof(int))
        self.init_beams(k, self.length)

    cdef int init_beams(self, size_t k, size_t length) except -1:
        cdef TagState* parent
        cdef TagState* beam
        self.parents = <void**>malloc(k * sizeof(void*))
        self.beam = <void**>malloc(k * sizeof(void*))
        for i in range(k):
            parent = <TagState*>malloc(sizeof(TagState))
            parent.prev = NULL
            parent.score = 0
            parent.hist[0] = 0
            parent.hist[1] = 0
            parent.alt[0] = 0
            parent.alt[1] = 0
            self.parents[i] = <void*>parent
        for i in range(k):
            beam = <TagState*>malloc(sizeof(TagState))
            beam.score = 0
            beam.hist[0] = 0
            beam.hist[1] = 0
            beam.alt[0] = 0
            beam.alt[1] = 0
            beam.prev = NULL
            self.beam[i] = <void*>beam

    cdef int swap_beam(self):
        # TODO: Find a better scheme for this
        cdef void** parents = self.parents
        self.parents = self.beam
        self.beam = parents 
        cdef TagState* beam
        for i in range(self.k):
            copy = <TagState*>malloc(sizeof(TagState))
            orig = <TagState*>self.beam[i]
            copy.score = orig.score
            copy.hist[0] = orig.hist[0]
            copy.hist[1] = orig.hist[1]
            copy.prev = orig.prev
            self.beam[i] = <void*>copy
        self.psize = self.bsize
        self.bsize = 0 

    def __dealloc__(self):
        cdef TagState* s
        cdef TagState* prev
        cdef size_t addr
        to_free = set()
        for i in range(self.k):
            s = <TagState*>self.parents[i]
            addr = <size_t>self.parents[i]
            while addr not in to_free and s.prev != NULL:
                to_free.add(addr)
                s = s.prev
            s = <TagState*>self.beam[i]
            addr = <size_t>self.beam[i]
            while addr not in to_free and s.prev != NULL:
                to_free.add(<size_t>s)
                s = s.prev
            free(self.valid[i])
            free(self.costs[i])
        for addr in to_free:
            if addr != 0:
                s = <TagState*>addr
                free(s)
        free(self.parents)
        free(self.beam)
        free(self.valid)
        free(self.costs)

    cdef uint64_t extend_state(self, size_t parent_idx, size_t b_idx,
                               size_t clas, double score):
        parent = <TagState*>self.parents[parent_idx]
        # We've got two arrays of states, and we swap beam-for-parents.
        # So, s here will get manipulated, then copied into parents later.
        cdef TagState* s = <TagState*>self.beam[self.bsize]
        s.score = score
        s.prev = parent
        s.hist[1] = parent.hist[0]
        s.hist[0] = clas
        return (s.hist[0] * self.nr_class) * s.hist[1]

    cdef double get_score(self, size_t parent_idx):
        cdef TagState* s = <TagState*>self.parents[parent_idx]
        return  s.score

    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd, bint* edits) except -1:
        # No need to copy heads for root and start symbols
        s = <TagState*>self.beam[0]
        # TODO: Check this
        i = self.t
        while i != 0 and s.prev != NULL:
            i -= 1
            tags[i] = s.hist[0]
            s = s.prev

    cdef bint _is_finished(self, int p_or_b, size_t idx):
        return False

    #cdef int eval_beam(self, size_t* gold):
    #    cdef size_t i, w
    #    cdef TagState* s
    #    c = 0
    #    for w in range(1, self.t):
    #        for i in range(self.k):
    #            s = <TagState*>self.beam[i]
    #            if s.tags[w] == gold[w]:
    #                c += 1
    #                break
    #    return c

cdef int fill_hist(size_t* hist, TagState* s, int t) except -1:
    while t >= 0 and s.prev != NULL:
        hist[t] = s.hist[0]
        s = s.prev
        t -= 1



