# cython: profile=True

from libc.stdint cimport uint64_t
from libc.string cimport memcpy
from cymem.cymem cimport Pool

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
        return self.reverse.get(code, 'UNK')

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


_pos_idx = Index(['ROOT', 'NONE', 'OOB', 'UNK'])
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
