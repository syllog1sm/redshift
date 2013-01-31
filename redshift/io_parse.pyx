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

def set_special_pos(root_pos, none_pos, oob_pos):
    global ROOT_POS, NONE_POS, OOB_POS
    ROOT_POS = root_pos
    NONE_POS = none_pos
    OOB_POS = oob_pos

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
        s.brown4[i] = 0
        s.brown6[i] = 0
        s.parse.heads[i] = 0
        s.parse.labels[i] = 0
    for i in range(MAX_TRANSITIONS):
        s.parse.moves[i] = 0
    # Don't initialise this, as repairs may introduce unpredictable numbers of
    # moves
    s.parse.n_moves = 0
    s.parse.score = 1.0
    py_words = ['<oob>'] + py_words + ['<root>']
    py_tags = ['OOB'] + py_tags + ['ROOT']
    for i in range(length + 2):
        s.words[i] = index.hashes.encode_word(py_words[i])
        s.pos[i] = index.hashes.encode_pos(py_tags[i])
    return s


def read_conll(conll_str, moves=None):
    """reads a CoNLL dependency data file, returning (words,POS,governors)
    all of which are lists beginning with a dummy head word"""
    cdef:
        size_t i
        object words, tags, heads, labels, token_str, word, pos, head, label
        Sentences sentences
    sent_strs = conll_str.strip().split('\n\n')
    set_special_pos(index.hashes.encode_pos('ROOT'),
                         index.hashes.encode_pos('NONE'),
                         index.hashes.encode_pos('OOB'))
    sentences = Sentences(max_length=len(sent_strs))
    first_sent = sent_strs[0]
    cdef size_t id_
    for i, sent_str in enumerate(sent_strs):
        if sent_str == first_sent:
            id_ = 0
        else:
            id_ += 1
        words = []
        tags = []
        heads = []
        labels = []
        for token_str in sent_str.split('\n'):
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
            heads.append(int(head))
            labels.append(STR_TO_LABEL.get(label, 0))
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
        words = []
        tags = []
        for token_str in sent_str.split():
            word, pos = token_str.rsplit('/', 1)
            words.append(word)
            tags.append(pos)
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
                if heads[i] == -1:
                    s.parse.heads[i + 1] = s.length + 1
                    s.parse.labels[i + 1] = ROOT_LABEL
                else:
                    s.parse.heads[i + 1] = <size_t>(heads[i] + 1)
                    s.parse.labels[i + 1] = <size_t>labels[i]
        self.s[self.length] = s
        self.length += 1
        self.strings.append((words, tags))

    def add_moves(self, object all_moves):
        all_moves = all_moves.strip().split('\n\n')
        assert len(all_moves) == self.length
        for i, sent_moves in enumerate(all_moves):
            self.s[i].parse.n_moves = 0
            for j, move_str in enumerate(sent_moves.split('\n')):
                name, lmove_id = move_str.split()
                lmove_id = int(lmove_id)
                self.s[i].parse.moves[j] = lmove_id
                self.s[i].parse.n_moves += 1

    def write_moves(self, out_file):
        """Write the history of moves and scores"""
        cdef Sentence s
        cdef size_t i, j, move_and_label
        cdef object move
        for i in range(self.length):
            s = self.s[i]
            for j in range(s.parse.n_moves):
                line = u'%s\t%d\n' % (MOVE_STRS[s.parse.moves[j]], s.parse.moves[j])
                out_file.write(line)
            out_file.write(u'\n')

    def write_parses(self, out_file):
        cdef Sentence* s
        cdef size_t i, j
        for i in range(self.length):
            s = &self.s[i]
            py_words, py_pos = self.strings[i]
            for j in range(1, s.length + 1):
                if s.parse.heads[j] == s.length + 1:
                    head = -1
                else:
                    head = <int>(s.parse.heads[j] - 1)
                fields = (j, py_words[j - 1], py_pos[j - 1], head,
                          LABEL_STRS[s.parse.labels[j]])
                out_file.write(u'%d\t%s\t%s\t%s\t%s\n' % fields)
            out_file.write(u'\n')

    cdef evaluate(self, Sentences gold):
        cdef float nc = 0
        cdef float n = 1e-6
        assert self.length == gold.length
        for i in range(self.length):
            t = self.s[i]
            g = gold.s[i]
            for j in range(1, t.length + 1):
                if g.parse.labels[j] == PUNCT_LABEL:
                    continue
                nc += t.parse.heads[j] == g.parse.heads[j]
                n += 1
        return nc/n

    def __dealloc__(self):
        free(self.s)
        

    property length:
        def __get__(self): return self.length

    property scores:
        def __get__(self):
            return [p.score for p in self.parses]


def merge_sentences(object sents_list):
    cdef Sentences sents
    cdef size_t max_len = sum(sents.length for sents in sents_list)
    cdef Sentences merged = Sentences(max_len)
    cdef size_t n = 0
    for sents in sents_list:
        for i in range(sents.length):
            memcpy(&(merged.s[n]), &(sents.s[i]), sizeof(Sentence))
            merged.strings.append(sents.strings[i])
            n += 1
    return merged
