from libcpp.utility cimport pair
from libcpp.vector cimport vector

DEF VOCAB_SIZE = 1e6
DEF TAG_SET_SIZE = 100


cdef extern from "sparsehash/dense_hash_map" namespace "google":
    cdef cppclass dense_hash_map[K, D]:
        K& key_type
        D& data_type
        pair[K, D]& value_type
        size_t size_type
        cppclass iterator:
            pair[K, D]& operator*() nogil
            iterator operator++() nogil
            iterator operator--() nogil
            bint operator==(iterator) nogil
            bint operator!=(iterator) nogil
        iterator begin()
        iterator end()
        size_t size()
        size_t max_size()
        bint empty()
        size_t bucket_count()
        size_t bucket_size(size_t i)
        size_t bucket(K& key)
        double max_load_factor()
        void max_load_vactor(double new_grow)
        double min_load_factor()
        double min_load_factor(double new_grow)
        void set_resizing_parameters(double shrink, double grow)
        void resize(size_t n)
        void rehash(size_t n)
        dense_hash_map()
        dense_hash_map(size_t n)


        void swap(dense_hash_map&)
        pair[iterator, bint] insert(pair[K, D]) nogil
        void set_empty_key(K&)
        void set_deleted_key(K& key)
        void clear_deleted_key()
        void erase(iterator pos)
        size_t erase(K& k)
        void erase(iterator first, iterator last)
        void clear()
        void clear_no_resize()
        pair[iterator, iterator] equal_range(K& k)
        D& operator[](K&) nogil

cdef extern from "MurmurHash3.h":
    void MurmurHash3_x86_32(void * key, int len, int seed, void* out)
    void MurmurHash3_x86_128(void * key, int len, int seed, void* out)



cdef class Index:
    cdef object path
    cdef bint save_entries
    cdef object out_file
    cdef unsigned long i

    cpdef set_path(self, path)
    cpdef save_entry(self, int i, object feat_str, int hashed, int value)
    cpdef save(self)
    cpdef load(self, path)


cdef class StrIndex(Index):
    cdef dense_hash_map[int, int] table
    cdef unsigned long encode(self, char* feature) except 0
    cpdef load_entry(self, size_t i, object key, long hashed, long value)


cdef class FeatIndex(Index):
    cdef int n
    cdef int p_i
    cdef int threshold
    cdef bint count_features
    cdef vector[dense_hash_map[long, long]] tables
    cdef vector[dense_hash_map[long, long]] unpruned
    cdef dense_hash_map[long, long] freqs
    cdef unsigned long encode(self, size_t* feature, size_t length, size_t i)
    cpdef load_entry(self, size_t i, object key, long hashed, unsigned long value)


cdef unsigned long encode_feat(size_t* feature, size_t length, size_t i)


cdef class InstanceCounter:
    cdef int n
    cdef vector[dense_hash_map[long, long]] counts_by_class
    cdef long add(self, size_t class_, size_t sent_id, size_t* history,
                  bint freeze_count) except 0
