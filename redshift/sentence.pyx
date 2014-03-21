from libc.stdlib cimport malloc, calloc, free

cimport index.lexicon
cimport index.hashes
import index.lexicon
from index.lexicon cimport Lexeme

from index.hashes import encode_pos
from index.hashes import encode_label
from index.hashes import decode_pos
from index.hashes import decode_label

cdef Sentence* init_sent(list words_cn) except NULL:
    cdef Sentence* s = <Sentence*>malloc(sizeof(Sentence))
    s.n = len(words_cn)
    s.steps = <Step*>calloc(s.n, sizeof(Step))
    s.answer = <AnswerToken*>calloc(s.n, sizeof(AnswerToken))
    cdef Token t
    for i in range(len(words_cn)):
        init_step(words_cn[i], &s.steps[i])
        # TODO: Support passing in the answer for a lattice
        s.answer[i].word = 0
        t = words_cn[i][0][1]
        s.answer[i].tag = t.c_parse.tag

        s.answer[i].head = t.c_parse.head if t.c_parse.head != 0 else s.n - 1
        s.answer[i].label = t.c_parse.label
        s.answer[i].is_edit = t.c_parse.is_edit
        s.answer[i].is_break = t.c_parse.is_break
    s.answer[0].head = 0
    s.answer[s.n - 1].head = 0
    # We used to have to do this to ensure that the zero-position evaluated
    # to 0, for the feature calculation (0 is a special value indicating absent)
    # There's probably a better way, but for now...
    cdef Lexeme* w
    for i in range(s.steps[0].n):
        s.steps[0].nodes[i].orig = 0
        s.steps[0].nodes[i].norm = 0
    s.answer[0].tag = 0
    # For the output
    return s


cdef int init_step(list cn_step, Step* step) except -1:
    step.n = len(cn_step)
    step.nodes = <Lexeme**>calloc(step.n, sizeof(Lexeme*))
    step.probs = <double*>calloc(step.n, sizeof(double))
    cdef Token token
    for i, (p, token) in enumerate(cn_step):
        step.probs[i] = p
        step.nodes[i] = token.c_word


cdef void free_sent(Sentence* s):
    cdef size_t i, j
    for i in range(s.n):
        free(&s.answer[i])
        free_step(&s.steps[i])
    free(s.steps)
    free(s.answer)


cdef void free_step(Step* s):
    # TODO: When we pass in pointers to these from a central vocab, remove this
    # free
    cdef size_t i
    for i in range(s.n):
        free(s.nodes[i])
    free(s.nodes)
    free(s.probs)


cdef class Input:
    def __init__(self, list words_cn):
        # Pad lattice with start and end tokens
        start_tok = Token('<start>', 'EOL', 0, 'ERR')
        end_tok = Token('<end>', 'EOL', 0, 'ERR')
        words_cn.insert(0, [(1.0, start_tok)])
        words_cn.append([(1.0, end_tok)])
        self.c_sent = init_sent(words_cn)

    def __dealloc__(self):
        # TODO: Fix memory
        pass
        #free_sent(self.c_sent)

    property length:
        def __get__(self):
            return self.c_sent.n

    property words:
        def __get__(self):
            return [index.lexicon.get_str(<size_t>self.c_sent.steps[i].nodes[self.c_sent.answer[i].word])
                    for i in range(self.c_sent.n)]

    property tags:
        def __get__(self):
            return [decode_pos(self.c_sent.answer[i].tag)
                    for i in range(self.c_sent.n)]

    property heads:
        def __get__(self):
            return [self.c_sent.answer[i].head for i in range(self.c_sent.n)]

    property labels:
        def __get__(self):
            return [decode_label(self.c_sent.answer[i].label)
                    for i in range(self.c_sent.n)]

    property edits:
        def __get__(self):
            return [self.c_sent.answer[i].is_edit for i in range(self.c_sent.n)]

    property breaks:
        def __get__(self):
            return [self.c_sent.answer[i].is_break for i in range(self.c_sent.n)]

    @classmethod
    def from_tokens(cls, tokens):
        """
        Create sentence from a flat list of unambiguous tokens, instead of a lattice.
        Tokens should be a list of either:
        - String-like, supporting obj.encode('ascii')
        - Object-like, supporting:
        """
        if hasattr(tokens[0], 'encode'):
            tokens = [Token.from_str(t) for t in tokens]
        return cls([[(1.0, t)] for t in tokens])

    @classmethod
    def from_conll(cls, i, conll_str):
        tokens = []
        for line in conll_str.split('\n'):
            fields = line.split()
            word_id = int(fields[0]) - 1
            word = fields[1]
            pos = fields[3]
            feats = fields[5].split('|')
            is_edit = len(fields) >= 3 and fields[2] == '1'
            is_break = len(fields) >= 4 and fields[3] == '1'
            head = int(fields[6])
            label = fields[7]
            tokens.append(Token(word, pos, head, label, is_edit, is_break))
        return cls.from_tokens(tokens)

    def to_conll(self):
        lines = []
        for i in range(1, self.length - 1):
            lines.append(conll_line_from_answer(i, &self.c_sent.answer[i],
                         self.c_sent.steps))
        return '\n'.join(lines)


cdef bytes conll_line_from_answer(size_t i, AnswerToken* a, Step* lattice):
    cdef bytes word = index.lexicon.get_str(<size_t>lattice[i].nodes[a.word])
    if not word:
        word = b'-OOV-'
    feats = '-|-|-|-'
    return '\t'.join((str(i), word, '_', 'NN', 'NN', feats, 
                     str(a.head), decode_label(a.label), '_', '_'))


cdef class Token:
    def __init__(self, word, pos=None, head=None, label=None,
                 is_edit=False, is_break=False):
        self.c_word = <Lexeme*>index.lexicon.lookup(word)
        self.c_parse = <AnswerToken*>malloc(sizeof(AnswerToken))
        if pos is not None:
            self.c_parse.tag = encode_pos(pos)
        if head is not None:
            self.c_parse.head = head
        if label is not None:
            self.c_parse.label = encode_label(label)
        self.c_parse.is_edit = is_edit
        self.c_parse.is_break = is_break

    def __dealloc__(self):
        free(self.c_parse)

    def __repr__(self):
        return 'Token(%s)' % (index.lexicon.get_str(<size_t>self.c_word))

    @classmethod
    def from_str(cls, token_str):
        fields = token_str.rsplit('/', 1)
        if len(fields) < 2:
            fields.append(None)
        word, pos = fields
        return cls(word, pos=pos)


def get_labels(sents):
    tags = set()
    left_labels = set()
    right_labels = set()
    cdef size_t i
    cdef Input sent
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            tags.add(sent.c_sent.answer[j].tag)
            if sent.c_sent.answer[j].head > j:
                left_labels.add(sent.c_sent.answer[j].label)
            else:
                right_labels.add(sent.c_sent.answer[j].label)
    return tags, list(sorted(left_labels)), list(sorted(right_labels))
