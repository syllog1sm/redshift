# cython: profile=True

from libc.stdint cimport uint64_t
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy

import os.path


cdef class Index:
    def __init__(self, entries):
        self.i = 0
        self.table = {}
        self.reverse = {}
        for entry in entries:
            self.lookup(entry)

    cpdef size_t lookup(self, bytes entry):
        if entry in self.table:
            return self.table[entry]
        else:
            self.i += 1
            self.table[entry] = self.i
            self.reverse[self.i] = entry
            return self.i

    cpdef bytes get_str(self, size_t code):
        return self.reverse.get(code, '')

    cpdef save(self, path):
        with open(path, 'w') as out_file:
            entries = self.reverse.items()
            entries.sort()
            for i, entry in entries:
                out_file.write('%d\t%s\n' % (i, entry))

    cpdef load(self, path):
        for line in open(path):
            if not line.strip():
                continue
            i, entry = line.split()
            i = int(i)
            self.table[entry] = i
            self.reverse[i] = entry
        self.i = i + 1


cdef class ScoresCache:
    def __cinit__(self, size_t scores_size, size_t pool_size=10000):
        self._cache = dense_hash_map[uint64_t, size_t]()
        self._cache.set_empty_key(0)
        self._pool = <double**>malloc(pool_size * sizeof(double*))
        for i in range(pool_size):
            self._pool[i] = <double*>malloc(scores_size * sizeof(double))
        self.i = 0
        self.pool_size = pool_size
        self.scores_size = scores_size
        self.n_hit = 0
        self.n_miss = 0
        
    cdef double* lookup(self, size_t size, void* kernel, bint* is_hit):
        cdef double** resized
        cdef uint64_t hashed = MurmurHash64A(kernel, size, 0)
        # Mix with a second hash for extra security -- collisions hurt here!!
        hashed += MurmurHash64B(kernel, size, 1)
        cdef size_t addr = self._cache[hashed]
        if addr != 0:
            self.n_hit += 1
            is_hit[0] = True
            return <double*>addr
        else:
            if self.i == self.pool_size:
                self._resize(self.pool_size * 2)
            addr = <size_t>self._pool[self.i]
            self.i += 1
            self._cache[hashed] = addr
            self.n_miss += 1
            is_hit[0] = False
            return <double*>addr
    
    def flush(self):
        self.i = 0
        self._cache.clear_no_resize()

    cdef int _resize(self, size_t new_size):
        cdef size_t i
        self.pool_size = new_size
        resized = <double**>malloc(self.pool_size * sizeof(double*))
        memcpy(resized, self._pool, self.i * sizeof(double*))
        for i in range(self.i, self.pool_size):
            resized[i] = <double*>malloc(self.scores_size * sizeof(double))
        free(self._pool)
        self._pool = resized

    def __dealloc__(self):
        for i in range(self.pool_size):
            free(self._pool[i])
        free(self._pool)



_pos_idx = Index(['ROOT', 'NONE', 'OOB'])
_label_idx = Index(['ERR', 'ROOT', 'P', 'erased'])


def load_pos_idx(path):
    global _pos_idx
    _pos_idx.load(path)


def load_label_idx(path):
    global _label_idx
    print "Load labels", path
    _label_idx.load(path)


def save_pos_idx(path):
    global _pos_idx
    _pos_idx.save(path)

def save_label_idx(path):
    global _label_idx
    _label_idx.save(path)


cpdef encode_pos(object pos):
    global _pos_idx
    cdef Index idx = _pos_idx
    return idx.lookup(pos)


def decode_pos(size_t i):
    global _pos_idx
    return _pos_idx.get_str(i)

def get_nr_pos():
    global _pos_idx
    cdef Index idx = _pos_idx
    return idx.i + 1

def encode_label(object label):
    global _label_idx
    cdef Index idx = _label_idx
    if label.upper() == 'ROOT':
        label = 'ROOT'
    elif label.upper() == 'PUNCT':
        label = 'P'
    return idx.lookup(label)


def decode_label(size_t i):
    global _label_idx
    return _label_idx.get_str(i)

def is_root_label(label):
    global _label_idx
    if type(label) == str:
        return encode_label(label) == encode_label('ROOT')
    else:
        return label == encode_label('ROOT')
