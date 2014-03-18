from ext.murmurhash cimport *
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport uint64_t

import os.path



cdef Vocab _VOCAB = None

def load(): # Called in index/__init__.py
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab()


cpdef size_t lookup(bytes word):
    global _VOCAB
    return _VOCAB.lookup(word)


cpdef bytes get_str(size_t word):
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = Vocab()
    return _VOCAB.strings.get(word, '')


cdef class Vocab:
    def __cinit__(self, loc=None):
        self.words.set_empty_key(0)
        self.strings = {}
        cdef object line
        cdef size_t i, word_id, freq
        cdef float upper_pc, title_pc
        if loc is None:
            loc = os.path.join(os.path.dirname(__file__), 'browns.txt')
        print "Loading vocab from ", loc 
        cdef size_t w
        for line in open(loc):
            if not line.strip():
                continue
            pieces = line.split()
            cluster = int(pieces[0])
            #upper_pc = float(pieces[1])
            #title_pc = float(pieces[2])
            word = pieces[1]
            freq = int(pieces[2])
            #self[word] = <size_t>init_word(word, cluster, upper_pc, title_pc)
            w = <size_t>init_word(word, cluster, 0.0, 0.0)
            self.words[_hash_str(word)] = w
            self.strings[<size_t>w] = word

    def __dealloc__(self):
        cdef size_t word_addr
        for word_addr in self.values():
            free(<Word*>word_addr)

    cdef size_t lookup(self, bytes word):
        cdef uint64_t hashed = _hash_str(word)
        cdef size_t addr = self.words[hashed]
        if addr == 0:
            addr = <size_t>init_word(word, 0, 0.0, 0.0)
            self.words[hashed] = addr
            self.strings[addr] = word
        return addr


cpdef bytes normalize_word(word):
    if '-' in word and word[0] != '-':
        return b'!HYPHEN'
    elif word.isdigit() and len(word) == 4:
        return b'!YEAR'
    elif word[0].isdigit():
        return b'!DIGITS'
    else:
        return word.lower()
    

cdef Word* init_word(bytes py_word, size_t cluster,
                     float upper_pc, float title_pc) except NULL:
    cdef Word* word = <Word*>malloc(sizeof(Word))
    word.orig = _hash_str(py_word)
    word.norm = _hash_str(normalize_word(py_word))
    word.suffix = _hash_str(py_word[-3:])
    word.prefix = ord(py_word[0])
    
    # Cut points determined by maximum information gain
    word.oft_upper = upper_pc >= 0.05
    word.oft_title = title_pc >= 0.3
    word.non_alpha = not py_word.isalpha()
    # TODO: Fix cluster stuff
    word.cluster = cluster
    return word


cdef inline uint64_t _hash_str(bytes s):
    return MurmurHash64A(<char*>s, len(s), 0)
