from libc.stdlib cimport malloc, free
from libc.string cimport strcpy, memcpy
import index.hashes

from pathlib import Path

cimport _state
from features cimport set_n_labels
DEF MAX_SENT_LEN = 256
DEF MAX_TRANSITIONS = 256 * 2


# Ensure ERR is always specified first, so 0 remains null label
LABEL_STRS = []
MOVE_STRS = []
STR_TO_LABEL = {}
ROOT_LABEL = 1
PUNCT_LABEL = 2


def set_labels(name):
    global LABEL_STRS, STR_TO_LABEL, PUNCT_LABEL, ROOT_LABEL
    if name == 'MALT':
        LABEL_STRS.extend('ERR,ROOT,P,NMOD,VMOD,PMOD,SUB,OBJ,AMOD,VC,SBAR,PRD,DEP'.split(','))
    elif name == 'NONE':
        LABEL_STRS.extend(('ERR', 'ROOT', 'P'))
    elif name == 'Stanford':
        LABEL_STRS.extend('ERR,ROOT,P,abbrev,acomp,advcl,advmod,amod,appos,attr,aux,auxpass,cc,ccomp,complm,conj,cop,csubj,csubjpass,dep,det,dobj,expl,infmod,iobj,mark,mwe,neg,nn,npadvmod,nsubj,nsubjpass,num,number,parataxis,partmod,pcomp,pobj,poss,preconj,predet,prep,prt,ps,purpcl,quantmod,rcmod,rel,tmod,xcomp'.split(','))
    else:
        raise StandardError, "Unrecognised label set: %s" % name
    for i, label in enumerate(LABEL_STRS):
        STR_TO_LABEL[label] = i
    ROOT_LABEL = STR_TO_LABEL['ROOT']
    PUNCT_LABEL = STR_TO_LABEL['P']
    set_n_labels(len(LABEL_STRS))


def set_moves(moves):
    global MOVE_STRS
    for move in moves:
        MOVE_STRS.append(move)


cdef Sentence make_sentence(size_t id_, size_t length, object py_words, object py_tags):
    cdef:
        size_t i
        bytes py_word
        bytes py_pos
        char* raw_word
        char* raw_pos
        Sentence s
    s = Sentence(length=length, id=id_)
    for i in range(MAX_SENT_LEN):
        s.words[i] = 0
        s.pos[i] = 0
        s.browns[i] = 0
        s.parse.heads[i] = 0
        s.parse.labels[i] = 0
    for i in range(MAX_TRANSITIONS):
        s.parse.moves[i] = 0
        s.parse.move_labels[i] = 0
    # Don't initialise this, as repairs may introduce unpredictable numbers of
    # moves
    s.parse.n_moves = 0
    for i in range(length):
        s.words[i] = index.hashes.encode_word(py_words[i])
        s.pos[i] = index.hashes.encode_pos(py_tags[i])
    return s


def read_conll(conll_str, moves=None):
    cdef:
        size_t i
        object words, tags, heads, labels, token_str, word, pos, head, label
        Sentences sentences
    sent_strs = conll_str.strip().split('\n\n')
    sentences = Sentences(max_length=len(sent_strs))
    first_sent = sent_strs[0]
    cdef size_t id_
    for id_, sent_str in enumerate(sent_strs):
        if sent_str == first_sent:
            id_ = 0
        else:
            id_ += 1
        words = ['<start>']
        tags = ['OOB']
        heads = [0]
        labels = [0]
        token_strs = sent_str.split('\n')
        for token_str in token_strs:
            pieces = token_str.split()
            if len(pieces) == 5:
                pieces.pop(0)
            try:
                word, pos, head, label = pieces
            except:
                print pieces
                raise
            words.append(word)
            tags.append(pos)
            if head == '-1':
                head = len(token_strs)
            heads.append(int(head) + 1)
            labels.append(STR_TO_LABEL.get(label, 0))
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append(0)
        #for i, (word, head) in enumerate(zip(words, heads)):
        #    print i, word, head
        sentences.add(id_, words, tags, heads, labels)
    if moves is not None and moves.strip():
        sentences.add_moves(moves)
    return sentences

    
def read_pos(file_str):
    cdef:
        size_t i
        object token_str, word, pos, words, tags
        Sentences sentences

    sent_strs = file_str.strip().split('\n')
    sentences = Sentences(max_length=len(sent_strs))
    for i, sent_str in enumerate(sent_strs):
        words = ['<start>']
        tags = ['OOB']
        for token_str in sent_str.split():
            word, pos = token_str.rsplit('/', 1)
            words.append(word)
            tags.append(pos)
        words.append('<root>')
        tags.append('<oob>')
        tags.append('ROOT')
        tags.append('OOB')
        sentences.add(i, words, tags, None, None)
    return sentences


cdef class Sentences:
    def __cinit__(self, size_t max_length=100000):
        self.strings = []
        self.length = 0
        self.s = <Sentence*>malloc(sizeof(Sentence) * max_length)
        self.max_length = max_length

    cpdef int add(self, size_t id_, object words, object tags, object heads, object labels) except -1:
        cdef Sentence s
        cdef size_t n
        cdef size_t i
        n = len(words)
        s = make_sentence(id_, n, words, tags)
        # NB: This doesn't initialise moves
        if heads and labels:
            for i in range(s.length):
                s.parse.heads[i] = <size_t>heads[i]
                s.parse.labels[i] = <size_t>labels[i]
        self.s[self.length] = s
        self.length += 1
        self.strings.append((words, tags))

    def add_moves(self, object all_moves):
        # TODO: Update this for new format
        raise StandardError
        all_moves = all_moves.strip().split('\n\n')
        assert len(all_moves) == self.length
        for i, sent_moves in enumerate(all_moves):
            self.s[i].parse.n_moves = 0
            for j, move_str in enumerate(sent_moves.split('\n')):
                name, move_id, move_label = move_str.split()
                move_id = int(move_id)
                self.s[i].parse.moves[j] = int(move_id)
                self.s[i].parse.move_labels[j] = int(move_label)
                self.s[i].parse.n_moves += 1

    def write_moves(self, out_file):
        cdef Sentence s
        cdef size_t i, j, move_and_label
        cdef object move
        move_strs = ['E', 'S', 'D', 'R', 'L', 'W']
        for i in range(self.length):
            s = self.s[i]
            for j in range(s.parse.n_moves):
                # TODO: THis is quite a hack, because we shouldn't need the details
                # of how the move/labels are coupled here. But, shrug.
                paired = s.parse.moves[j]
                move = paired / len(LABEL_STRS)
                label = paired % len(LABEL_STRS)
                line = u'%d\t%s\t%s\n' % (paired, move_strs[move], LABEL_STRS[label])
                out_file.write(line)
            out_file.write(u'\n')

    def write_parses(self, out_file):
        cdef Sentence* s
        cdef size_t i, j
        for i in range(self.length):
            s = &self.s[i]
            py_words, py_pos = self.strings[i]
            for j in range(1, s.length - 1):
                if s.parse.heads[j] == (s.length - 1):
                    head = -1
                else:
                    head = <int>(s.parse.heads[j] - 1)
                fields = (j - 1, py_words[j], py_pos[j], head,
                          LABEL_STRS[s.parse.labels[j]])
                out_file.write(u'%d\t%s\t%s\t%s\t%s\n' % fields)
            out_file.write(u'\n')

    cdef evaluate(self, Sentences gold):
        cdef double nc = 0
        cdef double n = 0
        assert self.length == gold.length
        for i in range(self.length):
            t = self.s[i]
            g = gold.s[i]
            for j in range(1, t.length):
                if g.parse.labels[j] == PUNCT_LABEL:
                    continue
                nc += t.parse.heads[j] == g.parse.heads[j]
                n += 1
        return nc/n

    def __dealloc__(self):
        free(self.s)
        
    property length:
        def __get__(self): return self.length
