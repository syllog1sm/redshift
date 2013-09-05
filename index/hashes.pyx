# cython: profile=True

DEF VOCAB_SIZE = 1e6
DEF TAG_SET_SIZE = 100
DEF LABEL_SIZE = 200

from libc.stdint cimport uint64_t
from libc.stdlib cimport calloc, malloc, free
from libc.string cimport memcpy

import os.path

import data
    
cdef class StrIndex:
    def __cinit__(self, expected_size, uint64_t i=2):
        self.table.set_empty_key(0)
        self.table.resize(expected_size)
        self.i = i
        self.strings = {}
        self.case_stats = {}

    def load_case_stats(self, loc):
        for line in open(loc):
            if not line.strip():
                continue
            word, caps_freq, title_freq, lower_freq = line.split()
            if word == 'i':
                continue
            caps_freq = int(caps_freq)
            title_freq = int(title_freq)
            lower_freq = int(lower_freq)
            total = float(caps_freq + title_freq + lower_freq)
            self.case_stats[word] = (caps_freq / total, title_freq / total)

    cdef uint64_t encode(self, char* feature) except 0:
        cdef uint64_t value
        cdef uint64_t hashed = MurmurHash64A(<char*>feature, len(feature), 0)
        value = self.table[hashed]
        if value == 0:
            value = self.i
            self.table[hashed] = value
            self.i += 1
            self.strings[feature] = value
        return value

    def get_reverse_index(self):
        index = {}
        for s, i in self.strings.items():
            index[i] = s
        return index

    def load(self, path):
        cdef uint64_t hashed
        cdef uint64_t value
        cdef object feature
        max_i = 0
        for line in open(path):
            fields = line.strip().split()
            i = int(fields[0])
            feature = fields[1]
            hashed = MurmurHash64A(<char*>feature, len(feature), 0)
            self.table[hashed] = i
            self.strings[feature] = i
            if i > max_i:
                max_i = i
        self.i = max_i + 1

    def save(self, path):
        out = open(path, 'w')
        for string, value in sorted(self.strings.items(), key=lambda i: i[1]):
            out.write('%d\t%s\n' % (value, string))

    property vocab:
        def __get__(self):
            return self.vocab

    property case_stats:
        def __get__(self):
            return self.case_stats


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
_label_idx = StrIndex(LABEL_SIZE)
_cluster_idx = ClusterIndex()

_word_idx.load_case_stats(os.path.join(data.path, 'case_stats.txt'))
_cluster_idx.load(os.path.join(data.path, 'browns.txt'))


_pos_idx.encode('ROOT')
_pos_idx.encode('NONE')
_pos_idx.encode('OOB')


_label_idx.encode('ERR')
_label_idx.encode('ROOT')
_label_idx.encode('P')
_label_idx.encode('erased')


def load_idx(name, path):
    global _word_idx, _pos_idx, _label_idx
    if name == 'word':
        _word_idx.load(path)
    elif name == 'pos':
        _pos_idx.load(path)
    elif name == 'label':
        _label_idx.load(path)
    else:
        raise StandardError, name


def save_idx(name, path):
    global _word_idx, _pos_idx, _label_idx
    if name == 'word':
        _word_idx.save(path)
    elif name == 'pos':
        _pos_idx.save(path)
    elif name == 'label':
        _label_idx.save(path)


cpdef encode_word(object word):
    global _word_idx
    cdef StrIndex idx = _word_idx
    py_word = word.encode('ascii')
    raw_word = py_word
    return idx.encode(raw_word)


cpdef encode_pos(object pos):
    global _pos_idx
    cdef StrIndex idx = _pos_idx
    py_pos = pos.encode('ascii')
    raw_pos = py_pos
    a = idx.encode(raw_pos)
    return a


def encode_label(object label):
    global _label_idx
    cdef StrIndex idx = _label_idx
    if label.upper() == 'ROOT':
        label = 'ROOT'
    elif label.upper() == 'PUNCT':
        label = 'P'
    py_label = label.encode('ascii')
    raw_label = py_label
    return idx.encode(raw_label)


def reverse_pos_index():
    global _pos_idx
    cdef StrIndex idx = _pos_idx
    return idx.get_reverse_index()

def reverse_word_index():
    global _word_idx
    cdef StrIndex idx = _word_idx
    return idx.get_reverse_index()


def reverse_label_index():
    global _label_idx
    cdef StrIndex idx = _label_idx
    return idx.get_reverse_index()


def is_root_label(label):
    global _label_idx
    if type(label) == str:
        return encode_label(label) == encode_label('ROOT')
    else:
        return label == encode_label('ROOT')


cpdef int get_freq(object word) except -1:
    global _word_idx
    return _word_idx.vocab.get(str(word), 0)


def get_case_stats():
    global _word_idx
    return _word_idx.case_stats

def get_clusters():
    global _cluster_idx
    return _cluster_idx


def get_max_context():
    global _word_idx
    cdef StrIndex idx = _word_idx
    return idx.i + 1


cdef class ClusterIndex:
    def __cinit__(self, thresh=1, prefix_len=6):
        self.thresh = thresh
        self.prefix_len = prefix_len
        self.n = 0

    def load(self, loc):
        entries = [('1', encode_word('<root>'), 40000),
                   ('1', encode_word('<start>'), 40000)]
        cdef object line
        cdef size_t i, word_id, freq
        for line in open(loc):
            if not line.strip():
                continue
            pieces = line.split()
            cluster_str = pieces[0]
            word = pieces[1]
            freq = int(pieces[2])
            if freq >= self.thresh:
                encoded = encode_word(word)
                entries.append((cluster_str, encoded, freq))
                if encoded >= self.n:
                    self.n = encoded + 1
        self.table = <Cluster*>calloc(self.n, sizeof(Cluster))
        cdef Cluster* cluster
        cdef object c_str
        for cluster_str, word_id, freq in entries:
            cluster = &self.table[word_id]
            cluster.full = int(cluster_str, 2) + 1
            cluster.prefix4 = int(cluster_str[:4], 2) + 1
            cluster.prefix6 = int(cluster_str[:6], 2) + 1

    def __dealloc__(self):
        free(self.table)
