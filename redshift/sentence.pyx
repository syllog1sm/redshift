from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy
import index.hashes
cimport index.hashes


def get_labels(sents):
    tags = set()
    left_labels = set()
    right_labels = set()
    cdef size_t i
    cdef Sentence sent
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
 

cdef CParse* init_c_parse(size_t length, list word_ids, list heads, list labels,
                          list edits) except NULL:
    cdef CParse* parse = <CParse*>malloc(sizeof(CParse))
    parse.n_moves = 0
    parse.score = 0
    cdef size_t PADDING = 5
    cdef size_t size = length + PADDING
    parse.heads = <size_t*>calloc(size, sizeof(size_t))
    parse.labels = <size_t*>calloc(size, sizeof(size_t))
    parse.sbd = <size_t*>calloc(size, sizeof(size_t))
    parse.edits = <bint*>calloc(size, sizeof(bint))
    parse.moves = <size_t*>calloc(size * 2, sizeof(size_t))
    cdef size_t i
    cdef size_t segment = 0
    cdef size_t erased = index.hashes.encode_label('erased')
    for i in range(length):
        parse.heads[i] = <size_t>heads[i]
        parse.labels[i] = index.hashes.encode_label(labels[i])
        if i >= 1 and word_ids[i] is not None and word_ids[i - 1] >= word_ids[i]:
            segment += 1
        parse.sbd[i] = segment
        if edits:
            parse.edits[i] = <bint>edits[i]
            if parse.edits[i] and not edits[parse.heads[i]]:
                parse.labels[i] = erased
    return parse
 

cdef CSentence* init_c_sent(size_t id_, size_t length, py_word_ids, py_words,
        py_tags, py_heads, py_labels, py_edits) except NULL:
    cdef:
        size_t i
        CSentence* s
    cdef size_t PADDING = 5
    cdef size_t size = length + PADDING
    s = <CSentence*>malloc(sizeof(CSentence))
    s.length = length
    s.id = id_
    s.parse = init_c_parse(length, py_word_ids, py_heads, py_labels, py_edits)
    s.words = <size_t*>calloc(size, sizeof(size_t))
    s.pos = <size_t*>calloc(size, sizeof(size_t))       
    s.clusters = <size_t*>calloc(size, sizeof(size_t))
    s.cprefix4s = <size_t*>calloc(size, sizeof(size_t))
    s.cprefix6s = <size_t*>calloc(size, sizeof(size_t))
    s.suffix = <size_t*>calloc(size, sizeof(size_t))
    s.prefix = <size_t*>calloc(size, sizeof(size_t))
    s.oft_upper = <bint*>calloc(length, sizeof(bint))
    s.oft_title = <bint*>calloc(length, sizeof(bint))
    s.non_alpha = <bint*>calloc(length, sizeof(bint))
    cdef object word
    cdef dict case_dict = index.hashes.get_case_stats()
    cdef index.hashes.ClusterIndex brown_idx = index.hashes.get_clusters()
    for i in range(1, length):
        case_stats = case_dict.get(py_words[i])
        if case_stats is not None:
            upper_pc, title_pc = case_stats
            # Cut points determined by maximum information gain
            if upper_pc >= 0.05:
                s.oft_upper[i] = True
            if title_pc >= 0.3:
                s.oft_title[i] = True
        elif not py_words[i].isalpha():
            s.non_alpha[i] = True
        word = normalize_word(py_words[i])
        s.words[i] = index.hashes.encode_word(word)
        s.suffix[i] = index.hashes.encode_word(py_words[i][-3:])
        s.prefix[i] = index.hashes.encode_word(py_words[i][0])
        s.pos[i] = index.hashes.encode_pos(py_tags[i])
        #s.clusters[i] = brown_idx.table[s.words[i]].full
        #s.cprefix4s[i] = brown_idx.table[s.words[i]].prefix4
        #s.cprefix6s[i] = brown_idx.table[s.words[i]].prefix6
    return s


cdef free_sent(CSentence* s):
    free(s.words)
    free(s.pos)
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


cdef class Sentence:
    def __init__(self, size_t id_, tokens):
        n = len(tokens) + 2
        words = ['<start>']
        tags = ['OOB']
        heads = [0]
        labels = ['ERR']
        ids = [None]
        edits = [False]
        for token in tokens:
            words.append(token.word)
            tags.append(token.pos)
            edits.append(token.is_edit)
            heads.append(token.head if token.head != -1 else len(tokens))
            heads[-1] = heads[-1] + 1
            labels.append(token.label)
            ids.append(token.id)
        ids.append(None)
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append('ERR')
        edits.append(False)

        self.id = id_
        self.c_sent = init_c_sent(id_, n, ids, words, tags, heads, labels, edits)
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
            fields = (self.c_sent.parse.sbd[i+1], i, token.word, pos, head, label)
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
            word, pos, head, label = fields
            word_id = int(i)
            is_edit = False
        else:
            word_id, word, pos, pos2, feats, head, label = fields
            word_id = int(word_id)
            feats = feats.split('|')
            head = int(head) - 1
            if feats and feats[2] == '1':
                is_edit = True
            else:
                is_edit = False
        # For SWBD
        if pos.startswith('^'):
            pos = pos[1:]
            pos = pos.split('^')[0]
        return cls(word_id, word, pos, head, label, is_edit)
