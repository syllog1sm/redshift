from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy
import index.hashes

from pathlib import Path

DEF MAX_SENT_LEN = 400
DEF MAX_TRANSITIONS = 400 / 2

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
        LABEL_STRS.extend(('ERR', 'ROOT', 'P', 'NONE'))
    elif name == 'Stanford':
        LABEL_STRS.extend('ERR,ROOT,P,abbrev,acomp,advcl,advmod,amod,appos,attr,aux,auxpass,cc,ccomp,complm,conj,cop,csubj,csubjpass,dep,det,dobj,expl,infmod,iobj,mark,mwe,neg,nn,npadvmod,nsubj,nsubjpass,num,number,parataxis,partmod,pcomp,pobj,poss,preconj,predet,prep,prt,ps,purpcl,quantmod,rcmod,rel,tmod,xcomp'.split(','))
    elif name.endswith(".conll"):
        labels_set = set()
        for line in file(name):
           line = line.strip().split()
           if not line: continue
           labels_set.add(line[-3])
        LABEL_STRS.extend(labels_set)
    elif name.endswith(".malt"):
        labels_set = set()
        for line in file(name):
           line = line.strip().split()
           if not line: continue
           labels_set.add(line[-1])
        LABEL_STRS.extend(['ERR','ROOT','P'])
        LABEL_STRS.extend(labels_set)
    else:
        raise StandardError, "Unrecognised label set: %s" % name
    for i, label in enumerate(LABEL_STRS):
        STR_TO_LABEL[label] = i
    print "Loaded %s labels" % name
    ROOT_LABEL = STR_TO_LABEL['ROOT']
    PUNCT_LABEL = STR_TO_LABEL['P']
    return LABEL_STRS


def set_moves(moves):
    global MOVE_STRS
    for move in moves:
        MOVE_STRS.append(move)


cdef Sentence make_sentence(size_t id_, size_t length, py_ids, py_words, py_tags):
    cdef:
        size_t i
        bytes py_word
        bytes py_pos
        char* raw_word
        char* raw_pos
        Sentence s
    s = Sentence(length=length, id=id_)

    s.browns = <size_t*>calloc(length, sizeof(size_t))
    s.parse.heads = <size_t*>calloc(length, sizeof(size_t))
    s.parse.labels = <size_t*>calloc(length, sizeof(size_t))
    s.parse.sbd = <bint*>calloc(length, sizeof(bint))
    s.parse.moves = <size_t*>calloc(length * 2, sizeof(size_t))
    
    s.words = <size_t*>calloc(length, sizeof(size_t))
    s.pos = <size_t*>calloc(length, sizeof(size_t))       
    s.ids = <size_t*>calloc(length, sizeof(size_t))
    s.parse.n_moves = 0
    py_ids.insert(0, 0)
    py_ids.append(0)
    for i in range(length):
        s.words[i] = index.hashes.encode_word(py_words[i])
        s.pos[i] = index.hashes.encode_pos(py_tags[i])
        # Offset by 1 for the dummy start symbol
        s.ids[i] = py_ids[i]
    return s


def read_conll(conll_str, moves=None):
    cdef:
        size_t i
        object words, tags, heads, labels, token_str, word, pos, head, label
        Sentences sentences
    sent_strs = conll_str.strip().split('\n\n')
    sentences = Sentences(max_length=len(sent_strs))
    first_sent = sent_strs[0]
    cdef size_t word_idx = 0
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
        ids = []
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
            ids.append(word_idx)
            word_idx += 1
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append(0)
        #for i, (word, head) in enumerate(zip(words, heads)):
        #    print i, word, head
        sentences.add(id_, ids, words, tags, heads, labels)
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
    cdef size_t w_id = 0
    for i, sent_str in enumerate(sent_strs):
        words = ['<start>']
        tags = ['OOB']
        ids = []
        for token_str in sent_str.split():
            word, pos = token_str.rsplit('/', 1)
            words.append(word)
            tags.append(pos)
            ids.append(w_id)
            w_id += 1
        words.append('<root>')
        tags.append('<oob>')
        tags.append('ROOT')
        tags.append('OOB')
        sentences.add(i, ids, words, tags, None, None)
    return sentences


cdef class Sentences:
    def __cinit__(self, size_t max_length=100000):
        self.strings = []
        self.length = 0
        self.s = <Sentence*>malloc(sizeof(Sentence) * max_length)
        self.max_length = max_length

    cpdef int add(self, size_t id_, ids, words, tags, heads, labels) except -1:
        cdef Sentence s
        cdef size_t n
        cdef size_t i
        n = len(words)
        s = make_sentence(id_, n, ids, words, tags)
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
        move_strs = ['E', 'S', 'D', 'R', 'L', 'W', 'V']
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
                if s.parse.sbd[j] or j == (s.length - 2):
                    out_file.write(u'\n')

    cdef evaluate(self, Sentences gold):
        cdef double nc = 0
        cdef double n = 0
        assert self.length == gold.length
        for i in range(self.length):
            t = self.s[i]
            g = gold.s[i]
            assert t.length == g.length
            # Don't evaluate the start or root symbols
            for j in range(1, t.length - 1):
                if g.parse.labels[j] == PUNCT_LABEL:
                    continue
                if t.parse.labels[j] == g.parse.labels[j] == ROOT_LABEL:
                    # Call it correct if points anywhere in next sentence
                    for w in range(j, t.parse.heads[j]):
                        if t.parse.sbd[w]:
                            nc += 1
                            break
                else:
                    nc += t.parse.heads[j] == g.parse.heads[j]
                n += 1
        return nc/n

    def connect_sentences(self, size_t n):
        """
        Remove sentence boundaries, forming groups of N sentences. The head of
        a root word is the root word of the next sentence. The last sentence is
        headed by the root token.
        """
        cdef:
            size_t i, j, offset, m_id, old_id, new_id, prev_head
            size_t* words
            size_t* pos
            size_t* browns
            size_t* heads
            size_t* labels
            bint* sbd
            Sentence* sent

        cdef size_t n_merged = (self.length / n) + 1
        cdef Sentence* merged = <Sentence*>malloc(sizeof(Sentence) * n_merged)
        new_strings = []
        m_id = 0
        for i in range(0, self.length, n):
            # Dummy start and dummy end symbols
            length = 2
            for j in range(n):
                if (i + j) >= (self.length - 1):
                    break
                length += (self.s[i + j].length - 2)

            ids = <size_t*>calloc(length, sizeof(size_t))
            words = <size_t*>calloc(length, sizeof(size_t))
            pos = <size_t*>calloc(length, sizeof(size_t))
            browns = <size_t*>calloc(length, sizeof(size_t))
            heads = <size_t*>calloc(length, sizeof(size_t))
            labels = <size_t*>calloc(length, sizeof(size_t))
            sbd = <bint*>calloc(length, sizeof(bint))
            # Dummy start symbol
            words[0] = self.s[i].words[0]
            pos[0] = self.s[i].pos[0]
            browns[0] = self.s[i].browns[0]
            prev_head = 0
            new_string = tuple()
            offset = 0
            for j in range(n):
                if (i + j) >= (self.length - 1):
                    break
                sent = &self.s[i + j]
                new_string += self.strings[i + j]
                # Skip the dummy start and end tokens
                for old_id in range(1, sent.length - 1):
                    new_id = old_id + offset
                    ids[new_id] = sent.ids[old_id]
                    words[new_id] = sent.words[old_id]
                    pos[new_id] = sent.pos[old_id]
                    browns[new_id] = sent.pos[old_id]
                    labels[new_id] = sent.parse.labels[old_id]
                    # When we hit another sentence, correct the head of the last
                    # sentence's root.
                    heads[new_id] = offset + sent.parse.heads[old_id]
                    if sent.parse.heads[old_id] == (sent.length - 1):
                        if prev_head != 0:
                            heads[prev_head] = new_id
                        prev_head = new_id
                sbd[new_id] = True
                offset += (sent.length - 2)
                free(sent.ids)
                free(sent.words)
                free(sent.pos)
                free(sent.browns)
                free(sent.parse.heads)
                free(sent.parse.labels)
                free(sent.parse.moves)
                free(sent.parse.sbd)
                # TODO
                #free(sent)
            # Dummy root symbol
            new_id += 1
            old_id += 1
            words[new_id] = sent.words[old_id]
            pos[new_id] = sent.pos[old_id]
            browns[new_id] = sent.browns[old_id]
            merged[m_id] = Sentence(id=m_id, length=length, ids=ids, words=words,
                                    pos=pos, browns=browns)
            merged[m_id].parse.heads = heads
            merged[m_id].parse.labels = labels
            merged[m_id].parse.sbd = sbd
            merged[m_id].parse.moves = <size_t*>calloc(length * 2, sizeof(size_t))
            merged[m_id].parse.n_moves = length * 2
            new_strings.append(new_string)
            new_stirng = tuple()
            m_id += 1
        self.s = merged
        self.length = m_id
        print 'done'

    def __dealloc__(self):
        free(self.s)
        
    property length:
        def __get__(self): return self.length
