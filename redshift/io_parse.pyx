from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy
import index.hashes
cimport index.hashes


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


cdef int add_parse(Sentence* sent, list heads, list labels, edits) except -1:
    for i in range(sent.length):
        sent.parse.heads[i] = <size_t>heads[i]
        sent.parse.labels[i] = index.hashes.encode_label(labels[i])
        if edits:
            sent.parse.edits[i] = <size_t>edits[i]


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
        labels = ['ERR']
        ids = [0]
        token_strs = sent_str.split('\n')
        edits = [False]
        for tok_id, token_str in enumerate(token_strs):
            pieces = token_str.split()
            if len(pieces) == 10:
                word = pieces[1]
                pos = pieces[3]
                head = pieces[6]
                label = pieces[7]
                head = int(head) - 1
                is_edit = pieces[9] == 'True'
            else:
                if len(pieces) == 5:
                    pieces.pop(0)
                word, pos, head, label = pieces
                head = int(head)
                is_edit = False
            # For SWBD
            if pos.startswith('^'):
                pos = pos[1:]
            pos = pos.split('^')[0]
            words.append(word)
            tags.append(pos)
            edits.append(is_edit)
            if head == -1:
                head = len(token_strs)
            heads.append(int(head) + 1)
            labels.append(label)
            ids.append(word_idx)
            word_idx += 1
        ids.append(0)
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append('ERR')
        edits.append(False)
        sent = make_sentence(id_, len(ids), ids, words, tags, vocab_thresh)
        add_parse(sent, heads, labels, edits)
        sentences.add(sent, words, tags)
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
        ids = [0]
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
        ids.append(0)
        sent = make_sentence(i, len(ids), ids, words, tags, vocab_thresh)
        sentences.add(sent, words, tags)
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

    cdef int add(self, Sentence* sent, words, tags) except -1:
        self.s[self.length] = sent
        self.length += 1
        self.strings.append((words, tags))

    def get_labels(self):
        tags = set()
        left_labels = set()
        right_labels = set()
        cdef size_t i
        cdef Sentence* sent
        for i in range(self.length):
            sent = self.s[i]
            for j in range(sent.length):
                tags.add(sent.pos[j])
                if sent.parse.heads[j] > j:
                    left_labels.add(sent.parse.labels[j])
                else:
                    right_labels.add(sent.parse.labels[j])
        return tags, left_labels, right_labels


    def write_parses(self, out_file):
        cdef Sentence* s
        cdef size_t i, j, w_id
        cdef int head
        pos_idx = index.hashes.reverse_pos_index()
        label_idx = index.hashes.reverse_label_index()
        for i in range(self.length):
            s = self.s[i]
            py_words, py_pos = self.strings[i]
            w_id = 0
            for j in range(1, s.length - 1):
                if index.hashes.is_root_label(s.parse.labels[j]):
                    head = -1
                else:
                    head = <int>(s.parse.heads[j]) - (j - w_id)
                fields = (w_id, py_words[j], pos_idx[s.pos[j]], head,
                          label_idx[s.parse.labels[j]],
                          str(bool(s.parse.edits[j])))
                out_file.write(u'%d\t%s\t%s\t%s\t%s\t%s\n' % fields)
                w_id += 1
                if s.parse.sbd[j] or j == (s.length - 2):
                    out_file.write(u'\n')
                    w_id = 0

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
