from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy
import index.hashes
cimport index.hashes


def get_labels(sents):
    tags = set()
    left_labels = set()
    right_labels = set()
    cdef size_t i
    cdef PySentence sent
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            tags.add(sent.c_sent.pos[j])
            if sent.c_sent.parse.heads[j] > j:
                left_labels.add(sent.c_sent.parse.labels[j])
            else:
                right_labels.add(sent.c_sent.parse.labels[j])
    return tags, list(sorted(left_labels)), list(sorted(right_labels))


def normalize_word(word):
    if '-' in word and word[0] != '-':
        return '!HYPHEN'
    elif word.isdigit() and len(word) == 4:
        return '!YEAR'
    elif word[0].isdigit():
        return '!DIGITS'
    else:
        return word.lower()
 

cdef Sentence* init_c_sent(size_t id_, size_t length, py_words, py_tags):
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
    s.parse.score = 0
    size = length + PADDING
    s.parse.heads = <size_t*>calloc(size, sizeof(size_t))
    s.parse.labels = <size_t*>calloc(size, sizeof(size_t))
    s.parse.sbd = <size_t*>calloc(size, sizeof(size_t))
    s.parse.edits = <bint*>calloc(size, sizeof(bint))
    s.parse.moves = <size_t*>calloc(size * 2, sizeof(size_t))
    
    s.words = <size_t*>calloc(size, sizeof(size_t))
    s.owords = <size_t*>calloc(size, sizeof(size_t))
    s.pos = <size_t*>calloc(size, sizeof(size_t))       
    s.alt_pos = <size_t*>calloc(size, sizeof(size_t))
    s.clusters = <size_t*>calloc(size, sizeof(size_t))
    s.cprefix4s = <size_t*>calloc(size, sizeof(size_t))
    s.cprefix6s = <size_t*>calloc(size, sizeof(size_t))
    s.suffix = <size_t*>calloc(size, sizeof(size_t))
    s.prefix = <size_t*>calloc(size, sizeof(size_t))

    s.non_alpha = <bint*>calloc(size, sizeof(bint))
    s.oft_upper = <bint*>calloc(size, sizeof(bint))
    s.oft_title = <bint*>calloc(size, sizeof(bint))

    cdef index.hashes.ClusterIndex brown_idx = index.hashes.get_clusters()
    cdef dict case_dict = index.hashes.get_case_stats()
    mask_value = index.hashes.encode_word('<MASKED>')
    types = set()
    for i in range(length):
        s.owords[i] = index.hashes.encode_word(py_words[i])
        word = normalize_word(py_words[i])
        s.words[i] = index.hashes.encode_word(word)
        s.pos[i] = index.hashes.encode_pos(py_tags[i])
        case_stats = case_dict.get(py_words[i])
        if case_stats is None:
            if not py_words[i].isalpha():
                s.non_alpha[i] = True
        else:
            upper_pc, title_pc = case_stats
            # Cut points determined by maximum information gain
            if upper_pc >= 0.05:
                s.oft_upper[i] = True
            if title_pc >= 0.3:
                s.oft_title[i] = True
        if s.owords[i] < brown_idx.n:
            s.clusters[i] = brown_idx.table[s.owords[i]].full
            s.cprefix4s[i] = brown_idx.table[s.owords[i]].prefix4
            s.cprefix6s[i] = brown_idx.table[s.owords[i]].prefix6
        # Use POS tag to semi-smartly get ' disambiguation
        s.suffix[i] = index.hashes.encode_word(py_words[i][-3:])
        s.prefix[i] = index.hashes.encode_word(py_words[i][0])
    s.words[0] = 0
    s.pos[0] = 0
    return s


cdef int add_parse(Sentence* sent, list word_ids, list heads, list labels,
                   list edits) except -1:
    cdef size_t segment = 0
    for i in range(sent.length):
        sent.parse.heads[i] = <size_t>heads[i]
        sent.parse.labels[i] = index.hashes.encode_label(labels[i])
        if i >= 1 and word_ids[i] is not None and word_ids[i - 1] >= word_ids[i]:
            sent.parse.sbd[i-1] = 1
        sent.parse.sbd[i] = 0
        if edits:
            sent.parse.edits[i] = <bint>edits[i]
            if sent.parse.edits[i] and not edits[sent.parse.heads[i]]:
                sent.parse.labels[i] = index.hashes.encode_label('erased')


cdef free_sent(Sentence* s):
    free(s.words)
    free(s.owords)
    free(s.pos)
    free(s.alt_pos)
    free(s.clusters)
    free(s.cprefix4s)
    free(s.cprefix6s)
    free(s.suffix)
    free(s.prefix)
    free(s.oft_upper)
    free(s.oft_title)
    free(s.non_alpha)

    free(s.parse.heads)
    free(s.parse.labels)
    free(s.parse.sbd)
    free(s.parse.edits)
    free(s.parse.moves)
    free(s.parse)
    free(s)


cdef class PySentence:
    def __init__(self, size_t id_, tokens):
        n = len(tokens) + 2
        words = ['<start>']
        tags = ['OOB']
        heads = [0]
        labels = ['ERR']
        ids = [None]
        edits = [False]
        for i, token in enumerate(tokens):
            words.append(token.word)
            tags.append(token.pos)
            edits.append(token.is_edit)
            if token.head == -1:
                heads.append(n - 1)
            else:
                heads.append(token.head + 1)
            labels.append(token.label)
            ids.append(token.id)
        ids.append(None)
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append('ERR')
        edits.append(False)

        self.id = id_
        cdef Sentence* sent = init_c_sent(id_, n, words, tags)
        add_parse(sent, ids, heads, labels, edits)
        self.c_sent = sent
        self.length = n
        self.tokens = tokens

    def __dealloc__(self):
        free_sent(self.c_sent)

    @classmethod
    def from_conll(cls, size_t id_, object sent_str):
        return cls(id_, [Token.from_str(i, t) for i, t in
                   enumerate(sent_str.split('\n'))])

    @classmethod
    def from_pos(cls, id_, object sent_str):
        return cls(id_, [Token.from_pos(i, *t.rsplit('/', 1)) for i, t in
                   enumerate(sent_str.strip().split(' '))])

    property length:
        def __get__(self): return self.length

    property tokens:
        def __get__(self): return self.tokens

    def to_conll(self):
        tokens = []
        label_idx = index.hashes.reverse_label_index()
        pos_idx = index.hashes.reverse_pos_index()
        for i, token in enumerate(self.tokens):
            if self.c_sent.parse.heads[i+1] == self.length - 1:
                head = -1
            else:
                head = <int>(self.c_sent.parse.heads[i+1] - 1)
            pos = pos_idx[self.c_sent.pos[i+1]]
            label = label_idx.get(self.c_sent.parse.labels[i+1], 'ERR')
            if self.c_sent.parse.edits[i + 1]:
                label = 'erased'
            sbd = 'T' if self.c_sent.parse.sbd[i + 1] else 'F'
            fields = (token.word, pos, head, label, sbd)
            tokens.append('\t'.join([str(f) for f in fields]))
        return '\n'.join(tokens)


class Token(object):
    def __init__(self, id_, word, pos, head, label, is_edit):
        self.id = id_
        self.word = word
        self.pos = pos
        self.head = int(head)
        self.label = label
        self.is_edit = is_edit

    @classmethod
    def from_pos(cls, id_, word, pos):
        return cls(id_, word, pos, 0, 'ERR', False)

    @classmethod
    def from_str(cls, i, token_str):
        fields = token_str.split()
        if len(fields) == 4:
            fields.append(False)
        if len(fields) == 5:
            word, pos, head, label, sbd = fields
            word_id = int(i)
            head = int(head)
            is_edit = head == i or label == 'erased'
            sbd = sbd == 'True'
        else:
            word_id, word, _, pos, pos2, feats, head, label, _, _ = fields
            word_id = int(word_id)
            feats = feats.split('|')
            head = int(head) - 1
            if len(feats) >= 3 and feats[2] == '1':
                is_edit = True
            else:
                is_edit = False
        # For SWBD
        if pos.startswith('^'):
            pos = pos[1:]
            pos = pos.split('^')[0]
        return cls(word_id, word, pos, head, label, is_edit)
