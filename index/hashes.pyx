# cython: profile=True
from pathlib import Path

DEF VOCAB_SIZE = 1e6
DEF TAG_SET_SIZE = 100

from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


cdef class Index:
    cpdef set_path(self, path):
        self.path = path
        self.out_file = path.open('w')
        self.save_entries = True

    cpdef save(self):
        if self.save_entries:
            self.out_file.close()
        self.save_entries = False

    cpdef save_entry(self, int i, object feat_str, object hashed, object value):
        self.out_file.write(u'%d\t%s\t%d\t%d\n' % (i, feat_str, hashed, value))

    cpdef load(self, path):
        cdef object hashed
        cdef uint64_t value
        for line in path.open():
            fields = line.strip().split()
            i = int(fields[0])
            key = fields[1]
            hashed = int(fields[2])
            value = int(fields[3])
            self.load_entry(i, key, hashed, value)


cdef class StrIndex(Index):
    def __cinit__(self, expected_size, uint64_t i=1):
        self.table.set_empty_key(0)
        self.table.resize(expected_size)
        self.i = i
        self.save_entries = False
    
    cdef uint64_t encode(self, char* feature) except 0:
        cdef uint64_t value
        cdef uint64_t hashed = MurmurHash64A(<char*>feature, len(feature), 0)
        value = self.table[hashed]
        if value == 0:
            value = self.i
            self.table[hashed] = value
            self.i += 1
            if self.save_entries:
                self.save_entry(0, str(feature), hashed, value)
        assert value < 1000000
        return value

    cpdef load_entry(self, uint64_t i, object key, uint64_t hashed, uint64_t value):
        self.table[hashed] = value

    def __dealloc__(self):
        if self.save_entries:
            self.out_file.close()

cdef class PruningFeatIndex(Index):
    def __cinit__(self):
        cdef dense_hash_map[uint64_t, uint64_t] *table
        cdef dense_hash_map[uint64_t, uint64_t] *pruned
        self.unpruned = vector[dense_hash_map[uint64_t, uint64_t]]()
        self.tables = vector[dense_hash_map[uint64_t, uint64_t]]()
        self.freqs = dense_hash_map[uint64_t, uint64_t]()
        self.i = 1
        self.p_i = 1

    cpdef set_path(self, path):
        self.path = path
        self.out_file = open(str(path), 'w')
        self.save_entries = True

    def set_n_predicates(self, uint64_t n):
        cdef uint64_t i
        self.n = n
        self.save_entries = False
        cdef uint64_t zero = 0
        for i in range(n):
            table = new dense_hash_map[uint64_t, uint64_t]()
            self.unpruned.push_back(table[0])
            self.unpruned[i].set_empty_key(zero)
            pruned = new dense_hash_map[uint64_t, uint64_t]()
            self.tables.push_back(pruned[0])
            self.tables[i].set_empty_key(zero)
        self.freqs.set_empty_key(zero)
        self.count_features = False
    
    cdef uint64_t encode(self, uint64_t* feature, uint64_t length, uint64_t i):
        cdef uint64_t value
        cdef uint64_t hashed
        hashed = MurmurHash64A(feature, length * sizeof(uint64_t), i)
        if not self.count_features:
            return self.tables[i][hashed]
        value = self.unpruned[i][hashed]
        if value == 0:
            value = self.i
            self.unpruned[i][hashed] = value
            self.i += 1
        self.freqs[value] += 1
        if self.freqs[value] == self.threshold:
            self.tables[i][hashed] = self.p_i
            if self.save_entries:
                self.out_file.write('%d\t_\t%d\t%d\n' % (i, hashed, value))
            self.p_i += 1
        return value

    cpdef load_entry(self, uint64_t i, object key, uint64_t hashed, uint64_t value):
        self.tables[i][hashed] = value

    def __dealloc__(self):
        if self.save_entries:
            self.out_file.close()

    def set_feat_counting(self, count_feats):
        self.count_features = count_feats

    def set_threshold(self, uint64_t threshold):
        self.threshold = threshold


cdef class FeatIndex(Index):
    def __cinit__(self):
        cdef dense_hash_map[uint64_t, uint64_t] *table
        self.tables = vector[dense_hash_map[uint64_t, uint64_t]]()
        self.i = 1

    cpdef set_path(self, path):
        self.path = path
        self.out_file = open(str(path), 'w')
        self.save_entries = True

    def set_n_predicates(self, uint64_t n):
        cdef uint64_t i
        self.n = n
        self.save_entries = False
        cdef uint64_t zero = 0
        for i in range(n):
            table = new dense_hash_map[uint64_t, uint64_t]()
            self.tables.push_back(table[0])
            self.tables[i].set_empty_key(zero)
    
    cdef uint64_t encode(self, uint64_t* feature, uint64_t length, uint64_t i):
        cdef uint64_t value
        cdef uint64_t hashed
        hashed = MurmurHash64A(feature, length * sizeof(uint64_t), i)
        value = self.tables[i][hashed]
        if value == 0:
            value = self.i
            self.tables[i][hashed] = value
            if self.save_entries:
                self.out_file.write('%d\t_\t%d\t%d\n' % (i, hashed, value))
            self.i += 1
        return value

    cpdef load_entry(self, uint64_t i, object key, uint64_t hashed, uint64_t value):
        self.tables[i][hashed] = value

    cpdef save(self):
        if self.save_entries:
            self.out_file.close()
        self.save_entries = False


    def __dealloc__(self):
        if self.save_entries:
            self.out_file.close()

    def set_feat_counting(self, count_feats):
        self.count_features = count_feats

    def set_threshold(self, uint64_t threshold):
        self.threshold = threshold


cdef class ScoresCache:
    def __cinit__(self, size_t scores_size, size_t pool_size=500):
        self._cache = dense_hash_map[uint64_t, size_t]()
        self._cache.set_empty_key(0)
        self._pool = <double**>malloc(pool_size * sizeof(double*))
        for i in range(pool_size):
            self._pool[i] = <double*>malloc(scores_size * sizeof(double))
        self.i = 0
        self.pool_size = pool_size
        self.scores_size = scores_size
        
    cdef double* lookup(self, size_t size, void* kernel, bint* is_hit):
        cdef double** resized
        cdef uint64_t hashed = MurmurHash64A(kernel, size, 0)
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
        print "Resizing cache to %d" % new_size
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


cdef class InstanceCounter:
    def __cinit__(self):
        self.n = 0
        self.counts_by_class = vector[dense_hash_map[long, long]]()

    cdef uint64_t add(self, uint64_t class_, uint64_t sent_id,
                  uint64_t* history, bint freeze_count) except 0:
        cdef long hashed = 0
        cdef dense_hash_map[long, long] *counts
        py_moves = []
        i = 0
        while history[i] != 0:
            py_moves.append(history[i])
            i += 1
        py_moves.append(sent_id)
        hashed = hash(tuple(py_moves))
        while class_ >= self.n:
            counts = new dense_hash_map[long, long]()
            self.counts_by_class.push_back(counts[0])
            self.counts_by_class[self.n].set_empty_key(0)
            self.n += 1
        assert hashed != 0
        if not freeze_count:
            self.counts_by_class[class_][hashed] += 1
            freq = self.counts_by_class[class_][hashed]
        else:
            freq = self.counts_by_class[class_][hashed]
            if freq == 0:
                freq = 1
            self.counts_by_class[class_][hashed] = -1
        return freq




_pos_idx = StrIndex(TAG_SET_SIZE)
_word_idx = StrIndex(VOCAB_SIZE, i=TAG_SET_SIZE)
_feat_idx = FeatIndex()

def init_feat_idx(int n, path):
    global _feat_idx
    _feat_idx.set_n_predicates(n)
    _feat_idx.set_path(path)

def init_word_idx(path):
    global _word_idx
    _word_idx.set_path(path)

def init_pos_idx(path):
    global _pos_idx
    _pos_idx.set_path(path)
    encode_pos('ROOT')
    encode_pos('NONE')
    encode_pos('OOB')


def load_feat_idx(n, path):
    global _feat_idx
    _feat_idx.set_n_predicates(n)
    _feat_idx.load(path)


def load_word_idx(path):
    global _word_idx
    _word_idx.load(path)

def load_pos_idx(path):
    global _pos_idx
    _pos_idx.load(path)

def set_feat_counting(bint feat_counting):
    global _feat_idx
    _feat_idx.set_feat_counting(feat_counting)

def set_feat_threshold(int threshold):
    global _feat_idx
    _feat_idx.set_threshold(threshold)

def encode_word(object word):
    global _word_idx
    cdef StrIndex idx = _word_idx
    py_word = word.encode('ascii')
    raw_word = py_word
    return idx.encode(raw_word)

def encode_pos(object pos):
    global _pos_idx
    cdef StrIndex idx = _pos_idx
    py_pos = pos.encode('ascii')
    raw_pos = py_pos
    return idx.encode(raw_pos)

cdef uint64_t encode_feat(uint64_t* feature, uint64_t length, uint64_t i):
    global _feat_idx
    cdef FeatIndex idx = _feat_idx
    return idx.encode(feature, length, i)

cdef FeatIndex get_feat_idx():
    return _feat_idx
