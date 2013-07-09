from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy
import index.hashes
cimport index.hashes

from pathlib import Path

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
        LABEL_STRS.extend('ERR,ROOT,P,abbrev,acomp,advcl,advmod,amod,appos,attr,'
                          'aux,auxpass,cc,ccomp,complm,conj,cop,csubj,csubjpass,'
                          'dep,det,dobj,expl,infmod,iobj,mark,mwe,neg,nn,npadvmod,'
                          'nsubj,nsubjpass,num,number,parataxis,partmod,pcomp,pobj,'
                          'poss,possessive,preconj,predet,prep,prt,ps,purpcl,quantmod,rcmod,'
                          'rel,tmod,xcomp,discourse,erased,parataxis,'
                          'cop,goeswith'.split(','))
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


cdef Sentence* make_sentence(size_t id_, size_t length, py_ids, py_words, py_tags,
                             size_t thresh):
    cdef:
        size_t i
        bytes py_word
        bytes py_pos
        char* raw_word
        char* raw_pos
        Sentence* s
    cdef size_t PADDING = 5
    s = <Sentence*>malloc(sizeof(Sentence))
    s.length = length
    s.id = id_
    s.parse = <_Parse*>malloc(sizeof(_Parse))
    s.parse.n_moves = 0
    size = length + PADDING
    s.parse.heads = <size_t*>calloc(size, sizeof(size_t))
    s.parse.labels = <size_t*>calloc(size, sizeof(size_t))
    s.parse.sbd = <bint*>calloc(size, sizeof(bint))
    s.parse.edits = <bint*>calloc(size, sizeof(bint))
    s.parse.moves = <size_t*>calloc(size * 2, sizeof(size_t))
    
    s.words = <size_t*>calloc(size, sizeof(size_t))
    s.owords = <size_t*>calloc(size, sizeof(size_t))
    s.pos = <size_t*>calloc(size, sizeof(size_t))       
    s.ids = <size_t*>calloc(size, sizeof(size_t))
    s.clusters = <size_t*>calloc(size, sizeof(size_t))
    s.cprefix4s = <size_t*>calloc(size, sizeof(size_t))
    s.cprefix6s = <size_t*>calloc(size, sizeof(size_t))
    s.orths = <size_t*>calloc(size, sizeof(size_t))
    s.parens = <size_t*>calloc(size, sizeof(size_t))
    s.quotes = <size_t*>calloc(size, sizeof(size_t))

    cdef index.hashes.ClusterIndex brown_idx = index.hashes.get_clusters()
    py_ids.insert(0, 0)
    py_ids.append(0)
    mask_value = index.hashes.get_mask_value()
    cdef size_t paren_cnt = 0
    cdef size_t quote_cnt = 0
    for i in range(length):
        s.words[i] = index.hashes.encode_word(py_words[i])
        s.owords[i] = s.words[i]
        s.pos[i] = index.hashes.encode_pos(py_tags[i])
        if s.words[i] < brown_idx.n:
            s.clusters[i] = brown_idx.table[s.words[i]].full
            s.cprefix4s[i] = brown_idx.table[s.words[i]].prefix4
            s.cprefix6s[i] = brown_idx.table[s.words[i]].prefix6
        if thresh != 0 and index.hashes.get_freq(py_words[i]) <= thresh:
            s.words[i] = mask_value
        s.ids[i] = py_ids[i]
        # Use POS tag to semi-smartly get ' disambiguation
        if py_tags[i] == "``":
            quote_cnt += 1
        elif py_tags[i] == "''":
            quote_cnt -= 1
        elif py_words[i] == "(" or py_words[i] == "[" or py_words[i] == "{":
            paren_cnt += 1
        elif py_words[i] == ")" or py_words[i] == "]" or py_words[i] == "}":
            paren_cnt -= 1
        s.orths[i] = ord(py_words[i][0])
        s.parens[i] = paren_cnt
        s.quotes[i] = quote_cnt
    return s

cdef free_sent(Sentence* s):
   
    free(s.words)
    free(s.owords)
    free(s.pos)
    free(s.ids)
    free(s.clusters)
    free(s.cprefix4s)
    free(s.cprefix6s)
    free(s.orths)
    free(s.parens)
    free(s.quotes)

    free(s.parse.heads)
    free(s.parse.labels)
    free(s.parse.sbd)
    free(s.parse.edits)
    free(s.parse.moves)
    free(s.parse)
    free(s)


def read_conll(conll_str, moves=None, vocab_thresh=0):
    cdef:
        size_t i
        object words, tags, heads, labels, token_str, word, pos, head, label
        Sentences sentences
    sent_strs = conll_str.strip().split('\n\n')
    sentences = Sentences(max_length=len(sent_strs), vocab_thresh=vocab_thresh)
    first_sent = sent_strs[0]
    cdef size_t word_idx = 0
    cdef size_t id_
    for id_, sent_str in enumerate(sent_strs):
        words = ['<start>']
        tags = ['OOB']
        heads = [0]
        labels = [0]
        token_strs = sent_str.split('\n')
        ids = []
        edits = [False]
        for tok_id, token_str in enumerate(token_strs):
            pieces = token_str.split()
            if len(pieces) == 10:
                word = pieces[1]
                pos = pieces[3]
                head = pieces[6]
                label = pieces[7]
                head = str(int(head) - 1)
                is_edit = pieces[9] == 'True'
            else:
                if len(pieces) == 5:
                    pieces.pop(0)
                try:
                    word, pos, head, label = pieces
                    is_edit = False
                except:
                    print pieces
                    raise
            # For SWBD
            if pos.startswith('^'):
                pos = pos[1:]
            pos = pos.split('^')[0]
            words.append(word)
            tags.append(pos)
            edits.append(is_edit)
            if head == '-1':
                head = len(token_strs)
            heads.append(int(head) + 1)
            if label.upper() == 'ROOT':
                label = 'ROOT'
            elif label.upper() == 'PUNCT':
                label = 'P'
            labels.append(STR_TO_LABEL.get(label, 0))
            ids.append(word_idx)
            word_idx += 1
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append(0)
        edits.append(False)
        sentences.add(id_, ids, words, tags, heads, labels, edits)
    if moves is not None and moves.strip():
        sentences.add_moves(moves)
    return sentences

    
def read_pos(file_str, vocab_thresh=0):
    cdef:
        size_t i
        object token_str, word, pos, words, tags
        Sentences sentences

    sent_strs = file_str.strip().split('\n')
    sentences = Sentences(max_length=len(sent_strs), vocab_thresh=vocab_thresh)
    cdef size_t w_id = 0
    for i, sent_str in enumerate(sent_strs):
        words = ['<start>']
        tags = ['OOB']
        ids = []
        for token_str in sent_str.split():
            try:
                word, pos = token_str.rsplit('/', 1)
            except:
                print sent_str
                print token_str
                raise
            # For SWBD
            pos = pos.split('^')[-1]
            words.append(word)
            tags.append(pos)
            ids.append(w_id)
            w_id += 1
        words.append('<root>')
        tags.append('ROOT')
        sentences.add(i, ids, words, tags, None, None, None)
    return sentences


cdef class Sentences:
    def __cinit__(self, size_t max_length=100000, vocab_thresh=0):
        print "Vocab thresh=%d" % vocab_thresh
        self.strings = []
        self.length = 0
        self.s = <Sentence**>malloc(sizeof(Sentence*) * max_length)
        self.max_length = max_length
        self.vocab_thresh = vocab_thresh

    def __dealloc__(self):
        for i in range(self.length):
            free_sent(self.s[i])
        free(self.s) 

    def get_labels(self):
        """Get the set of tags, left labels and right labels in the data"""
        seen_l_labels = set([])
        seen_r_labels = set([])
        seen_tags = set([1, 2, 3])
        for i in range(sents.length):
            sent = sents.s[i]
            for j in range(1, sent.length - 1):
                seen_tags.add(sent.pos[j])
                label = sent.parse.labels[j]
                if sent.parse.heads[j] > j:
                    seen_l_labels.add(label)
                else:
                    seen_r_labels.add(label)
        return seen_tags, seen_l_labels, seen_r_labels

    cpdef int add(self, size_t id_, ids, words, tags, heads, labels, edits) except -1:
        cdef Sentence* s
        cdef size_t n
        cdef size_t i
        n = len(words)
        s = make_sentence(id_, n, ids, words, tags, self.vocab_thresh)
        # NB: This doesn't initialise moves, or set sentence boundaries
        if heads and labels:
            for i in range(s.length):
                s.parse.heads[i] = <size_t>heads[i]
                s.parse.labels[i] = <size_t>labels[i]
                if edits:
                    s.parse.edits[i] = <bint>edits[i]
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
        cdef Sentence* s
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
        cdef size_t i, j, w_id
        cdef int head
        pos_idx = index.hashes.reverse_pos_index()
        for i in range(self.length):
            s = self.s[i]
            py_words, py_pos = self.strings[i]
            w_id = 0
            for j in range(1, s.length - 1):
                if s.parse.labels[j] == ROOT_LABEL:
                    head = -1
                else:
                    head = <int>(s.parse.heads[j]) - (j - w_id)
                try:
                    fields = (w_id, py_words[j], pos_idx[s.pos[j]], head,
                              LABEL_STRS[s.parse.labels[j]], str(bool(s.parse.edits[j])))
                except:
                    print j, s.length, s.parse.labels[j], len(py_words), len(py_pos)
                    raise
                out_file.write(u'%d\t%s\t%s\t%s\t%s\t%s\n' % fields)
                w_id += 1
                if s.parse.sbd[j] or j == (s.length - 2):
                    out_file.write(u'\n')
                    w_id = 0

    def evaluate(self, Sentences gold):
        def gen_words(Sentences sents):
            cdef:
                size_t* ids
                size_t* heads
                size_t* labels
                bint* sbd
            for i in range(sents.length):
                ids = sents.s[i].ids
                heads = sents.s[i].parse.heads
                labels = sents.s[i].parse.labels
                sbd = sents.s[i].parse.sbd
                for j in range(1, sents.s[i].length - 1):
                    yield j, ids[j], ids[heads[j]], labels[j], sbd[j]
        cdef Sentence* t_sent
        cdef Sentence* g_sent
        cdef double nc = 0
        cdef double n = 0
        cdef double sbd_nc = 0
        cdef double sbd_n = 0
        for t_tok, g_tok in zip(gen_words(self), gen_words(gold)):
            g_w, g_id, g_head, g_label, g_sbd = g_tok
            t_w, t_id, t_head, t_label, t_sbd = t_tok
            sbd_nc += g_sbd == t_sbd
            sbd_n += 1
            if g_label == PUNCT_LABEL:
                continue
            if g_label == ROOT_LABEL and t_label == ROOT_LABEL:
                nc += 1
            else:
                nc += g_head == t_head
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
            size_t* clusters
            size_t* cprefix4s
            size_t* cprefix6s
            size_t* heads
            size_t* labels
            size_t* orths
            size_t* parens
            size_t* quotes
            bint* sbd
            Sentence* sent
        raise StandardError
        if n == 0:
            n = self.length
        cdef size_t n_merged = (self.length / n) + 1
        cdef Sentence** merged = <Sentence**>malloc(sizeof(Sentence*) * n_merged)
        new_strings = []
        m_id = 0
        for i in range(0, self.length, n):
            # Dummy start and dummy end symbols
            length = 2
            new_words = [self.strings[0][0]]
            new_pos = [self.strings[1][0]]
            for j in range(n):
                if (i + j) >= (self.length - 1):
                    break
                length += (self.s[i + j].length - 2)
                new_words.extend(self.strings[i + j][0][1:-1])
                new_pos.extend(self.strings[i + j][1][1:-1])
            new_words.append(self.strings[0][-1])
            new_pos.append(self.strings[1][-1])
            new_strings.append((new_words, new_pos))

            ids = <size_t*>calloc(length, sizeof(size_t))
            words = <size_t*>calloc(length, sizeof(size_t))
            pos = <size_t*>calloc(length, sizeof(size_t))
            clusters = <size_t*>calloc(length, sizeof(size_t))
            cprefix4s = <size_t*>calloc(length, sizeof(size_t))
            heads = <size_t*>calloc(length, sizeof(size_t))
            labels = <size_t*>calloc(length, sizeof(size_t))
            orths = <size_t*>calloc(length, sizeof(size_t))
            parens = <size_t*>calloc(length, sizeof(size_t))
            quotes = <size_t*>calloc(length, sizeof(size_t))
            sbd = <bint*>calloc(length, sizeof(bint))
            # Dummy start symbol
            words[0] = self.s[i].words[0]
            pos[0] = self.s[i].pos[0]
            clusters[0] = self.s[i].clusters[0]
            cprefix4s[0] = self.s[i].cprefix4s[0]
            prev_head = 0
            offset = 0
            for j in range(n):
                if (i + j) >= (self.length - 1):
                    break
                sent = self.s[i + j]
                # Skip the dummy start and end tokens
                for old_id in range(1, sent.length - 1):
                    new_id = old_id + offset
                    ids[new_id] = sent.ids[old_id]
                    words[new_id] = sent.words[old_id]
                    pos[new_id] = sent.pos[old_id]
                    clusters[new_id] = sent.clusters[old_id]
                    cprefix4s[new_id] = sent.cprefix4s[old_id]
                    orths[new_id] = sent.orths[old_id]
                    parens[new_id] = sent.parens[old_id]
                    quotes[new_id] = sent.quotes[old_id]
                    labels[new_id] = sent.parse.labels[old_id]
                    sbd[new_id] = sent.parse.sbd[old_id]
                    # When we hit another sentence, correct the head of the last
                    # sentence's root.
                    heads[new_id] = offset + sent.parse.heads[old_id]
                    if sent.parse.heads[old_id] == (sent.length - 1):
                        if prev_head != 0:
                            heads[prev_head] = new_id
                        prev_head = new_id
                offset += (sent.length - 2)
                free_sent(sent)
            # Dummy root symbol
            new_id += 1
            old_id += 1
            words[new_id] = sent.words[old_id]
            pos[new_id] = sent.pos[old_id]
            clusters[new_id] = sent.clusters[old_id]
            cprefix4s[new_id] = sent.cprefix4s[old_id]
            ids[new_id] = sent.ids[old_id]
            merged[m_id] = <Sentence*>malloc(sizeof(Sentence))
            merged[m_id].id = m_id
            merged[m_id].length = length
            merged[m_id].parse = <_Parse*>malloc(sizeof(_Parse))
            merged[m_id].parse.n_moves = 0
            merged[m_id].ids = ids
            merged[m_id].words = words
            merged[m_id].pos = pos
            merged[m_id].clusters = clusters
            merged[m_id].cprefix4s = cprefix4s
            merged[m_id].orths = orths
            merged[m_id].parens = parens
            merged[m_id].quotes = quotes
            merged[m_id].parse.heads = heads
            merged[m_id].parse.labels = labels
            merged[m_id].parse.sbd = sbd
            merged[m_id].parse.moves = <size_t*>calloc(length * 2, sizeof(size_t))
            m_id += 1
        free(self.s)
        self.s = merged
        self.length = m_id
        self.strings = new_strings
       
    property length:
        def __get__(self): return self.length
