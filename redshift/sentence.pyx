from libc.stdlib cimport malloc, calloc, free

cimport index.lexicon
cimport index.hashes
import index.lexicon
from index.lexicon cimport Lexeme
from index.lexicon cimport BLANK_WORD

from index.hashes import encode_pos
from index.hashes import encode_label
from index.hashes import decode_pos
from index.hashes import decode_label

cdef Sentence* init_sent(list words_lattice, list parse) except NULL:
    cdef Sentence* s = <Sentence*>malloc(sizeof(Sentence))
    s.n = len(words_lattice)
    s.lattice = <Step*>calloc(s.n, sizeof(Step))
    s.tokens = <Token*>calloc(s.n, sizeof(Token))
    cdef Token t
    for i in range(s.n):
        init_lattice_step(words_lattice[i], &s.lattice[i])
    cdef bint is_edit
    cdef bint is_break
    for i, (word_idx, tag, head, label, is_edit, is_break) in enumerate(parse):
        s.tokens[i].word = s.lattice[i].nodes[word_idx]
        if tag is not None:
            s.tokens[i].tag = index.hashes.encode_pos(tag) 
        if head is not None:
            s.tokens[i].head = head if head != 0 else s.n - 1
        if label is not None:
            s.tokens[i].label = index.hashes.encode_label(label)
        if is_edit is not None:
            s.tokens[i].is_edit = is_edit
        if is_break is not None:
            s.tokens[i].is_break = is_break
    # Set position 0 to be blank
    s.lattice[0].nodes[0] = &BLANK_WORD
    assert s.tokens[0].tag == 0
    assert s.tokens[0].head == 0
    assert s.tokens[0].label == 0
    return s


cdef int init_lattice_step(list lattice_step, Step* step) except -1:
    step.n = len(lattice_step)
    step.nodes = <Lexeme**>calloc(step.n, sizeof(Lexeme*))
    step.probs = <double*>calloc(step.n, sizeof(double))
    cdef size_t lex_addr
    for i, (p, word) in enumerate(lattice_step):
        step.probs[i] = p
        lex_addr = index.lexicon.lookup(word)
        step.nodes[i] = <Lexeme*>lex_addr


cdef void free_sent(Sentence* s):
    cdef size_t i, j
    for i in range(s.n):
        free(&s.tokens[i])
        free_step(&s.lattice[i])
    free(s.lattice)
    free(s.tokens)


cdef void free_step(Step* s):
    # TODO: When we pass in pointers to these from a central vocab, remove this
    # free
    cdef size_t i
    for i in range(s.n):
        free(s.nodes[i])
    free(s.nodes)
    free(s.probs)


cdef class Input:
    @classmethod
    def from_tokens(cls, tokens):
        """
        Create sentence from a flat list of unambiguous tokens, instead of a lattice.
        Tokens should be a list of (word, tag, head, label, is_edit, is_break)
        tuples
        """
        lattice = []
        parse = []
        for word, tag, head, label, is_edit, is_break in tokens:
            lattice.append([(1.0, word)])
            parse.append((0, tag, head, label, is_edit, is_break))
        return cls(lattice, parse)

    @classmethod
    def from_pos(cls, pos_strs):
        tokens = []
        for token_str in pos_strs:
            word, pos = token_str.rsplit('/', 1)
            tokens.append((word, pos, None, None, None, None))
        return cls.from_tokens(tokens)

    @classmethod
    def from_conll(cls, conll_str):
        tokens = []
        for line in conll_str.split('\n'):
            fields = line.split()
            word_id = int(fields[0]) - 1
            word = fields[1]
            pos = fields[3]
            feats = fields[5].split('|')
            is_edit = len(feats) >= 3 and feats[2] == '1'
            is_break = len(feats) >= 4 and feats[3] == '1'
            head = int(fields[6])
            label = fields[7]
            tokens.append((word, pos, head, label, is_edit, is_break))
        return cls.from_tokens(tokens)

    def __init__(self, list lattice, list parse):
        # Pad lattice with start and end tokens
        lattice.insert(0, [(1.0, '<start>')])
        parse.insert(0, (0, None, None, None, False, False))
        lattice.append([(1.0, '<end>')])
        parse.append((0, 'EOL', None, None, False, False))

        self.c_sent = init_sent(lattice, parse)

    def __dealloc__(self):
        # TODO: Fix memory
        pass
        #free_sent(self.c_sent)

    property length:
        def __get__(self):
            return self.c_sent.n

    property words:
        def __get__(self):
            return [index.lexicon.get_str(<size_t>self.c_sent.tokens[i].word)
                    for i in range(self.c_sent.n)]

    property tags:
        def __get__(self):
            return [decode_pos(self.c_sent.tokens[i].tag)
                    for i in range(self.c_sent.n)]

    property heads:
        def __get__(self):
            return [self.c_sent.tokens[i].head for i in range(self.c_sent.n)]

    property labels:
        def __get__(self):
            return [decode_label(self.c_sent.tokens[i].label)
                    for i in range(self.c_sent.n)]

    property edits:
        def __get__(self):
            return [self.c_sent.tokens[i].is_edit for i in range(self.c_sent.n)]

    property breaks:
        def __get__(self):
            return [self.c_sent.tokens[i].is_break for i in range(self.c_sent.n)]

    def to_conll(self):
        lines = []
        for i in range(1, self.length - 1):
            lines.append(conll_line_from_token(i, &self.c_sent.tokens[i],
                         self.c_sent.lattice))
        return '\n'.join(lines)


cdef bytes conll_line_from_token(size_t i, Token* a, Step* lattice):
    cdef bytes word = index.lexicon.get_str(<size_t>a.word)
    if not word:
        word = b'-OOV-'
    feats = '-|-|%d|-' % a.is_edit
    cdef bytes tag = index.hashes.decode_pos(a.tag)
    return '\t'.join((str(i), word, '_', tag, tag, feats, 
                     str(a.head), decode_label(a.label), '_', '_'))
