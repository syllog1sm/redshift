from _state cimport *
from transitions cimport TransitionSystem

from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair


cdef extern from "MurmurHash2.h":
    uint64_t MurmurHash64A(void * key, uint64_t len, int64_t seed)
    uint64_t MurmurHash64B(void * key, uint64_t len, int64_t seed)


cdef extern from "sparsehash/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        uint64_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin()
        iterator end()
        uint64_t size()
        uint64_t max_size()
        bint empty()
        uint64_t bucket_count()
        uint64_t bucket_size(uint64_t i)
        uint64_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(uint64_t n)
        void rehash(uint64_t n)
        dense_hash_map()
        dense_hash_map(uint64_t n)
        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        uint64_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil


cdef class Violation:
    """
    A gold/prediction pair where the g.score < p.score
    """
    cdef size_t t
    cdef size_t* ghist
    cdef size_t* phist
    cdef double delta
    cdef int cost
    cdef bint out_of_beam
    
    cdef int set(self, State*p, State* g, bint out_of_beam) except -1

cdef class Beam:
    cdef TransitionSystem trans
    cdef Violation violn
    #cdef priority_queue[pair[double, size_t]]* next_moves
    
    cdef State* gold
    cdef State** parents
    cdef State** beam
    cdef int** costs
    cdef bint** valid
    cdef object _prune_freqs
    
    cdef object upd_strat
    cdef size_t max_class
    cdef size_t k
    cdef size_t i
    cdef size_t t
    cdef size_t nr_skip
    cdef size_t length
    cdef size_t bsize
    cdef size_t psize
    cdef bint is_full
    cdef bint is_finished

    cdef Kernel* next_state(self, size_t i)
    cdef int cost_next(self, size_t i, size_t* tags, size_t* heads, size_t* labels) except -1
    cdef int extend_states(self, double** scores) except -1
    cdef bint check_violation(self)
    cdef int fill_parse(self, size_t* hist, size_t* tags, size_t* heads,
                        size_t* labels, bint* sbd) except -1

