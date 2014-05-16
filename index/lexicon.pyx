from ext.murmurhash cimport *
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport uint64_t

import os.path


BLANK_WORD = Lexeme(0, 0, 0, 0, False, False, False)

cdef Lexicon _LEXICON = None

def load(): # Called in index/__init__.py
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = Lexicon()


cpdef size_t lookup(bytes word):
    global _LEXICON
    return _LEXICON.lookup(word)


cpdef bytes get_str(size_t word):
    global _LEXICON
    if _LEXICON is None:
        _LEXICON = Lexicon()
    return _LEXICON.strings.get(word, '')


cdef class Lexicon:
    def __cinit__(self, loc=None):
        self.words.set_empty_key(0)
        self.strings = {}
        cdef object line
        cdef size_t i, word_id, freq
        cdef float upper_pc, title_pc
        if loc is None:
            loc = os.path.join(os.path.dirname(__file__), 'bllip-clusters')
        case_stats = {}
        for line in open(os.path.join(os.path.dirname(__file__), 'english.case')):
            word, upper, title = line.split()
            case_stats[word] = (float(upper), float(title))
        print "Loading vocab from ", loc 
        cdef size_t w
        for line in open(loc):
            cluster_str, word, freq_str = line.split()
            # Decode as a little-endian string, so that we can do & 15 to get
            # the first 4 bits. See _parse_features.pyx
            cluster = int(cluster_str[::-1], 2)
            #upper_pc = float(pieces[1])
            #title_pc = float(pieces[2])
            upper_pc, title_pc = case_stats.get(word.lower(), (0.0, 0.0))
            w = <size_t>init_word(word, cluster, upper_pc, title_pc)
            self.words[_hash_str(word)] = w
            self.strings[<size_t>w] = word

    def __dealloc__(self):
        cdef size_t word_addr
        for word_addr in self.values():
            free(<Lexeme*>word_addr)

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
    

cdef Lexeme* init_word(bytes py_word, size_t cluster,
                     float upper_pc, float title_pc) except NULL:
    cdef Lexeme* word = <Lexeme*>malloc(sizeof(Lexeme))
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
