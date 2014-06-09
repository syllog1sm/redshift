from libc.stdlib cimport malloc, calloc, free
from collections import namedtuple
import math

cimport index.lexicon
cimport index.hashes
import random
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
    assert s.n >= 3, words_lattice
    s.lattice = <Step*>calloc(s.n, sizeof(Step))
    s.tokens = <Token*>calloc(s.n, sizeof(Token))
    cdef Token t
    for i in range(s.n):
        init_lattice_step(words_lattice[i], &s.lattice[i])
    cdef bint is_edit
    for i, (word_idx, tag, head, label, sent_id, is_edit) in enumerate(parse):
        s.tokens[i].word = s.lattice[i].nodes[word_idx]
        assert s.tokens[i].word != NULL
        if tag is not None:
            s.tokens[i].tag = index.hashes.encode_pos(tag) 
        if head is not None:
            s.tokens[i].head = head if head != 0 else s.n - 1
        if label is not None:
            s.tokens[i].label = index.hashes.encode_label(label)
        if is_edit is not None:
            s.tokens[i].is_edit = is_edit
        if sent_id is not None:
            s.tokens[i].sent_id = sent_id
        # Initialise left edges to own value
        s.tokens[i].left_edge = i
    # Set left edges
    cdef size_t gov
    for i in range(1, s.n - 1):
        gov = s.tokens[i].head
        while i < s.tokens[gov].left_edge:
            s.tokens[gov].left_edge = i
            gov = s.tokens[gov].head
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
        if p == 0:
            p += 0.001
        step.probs[i] = math.log(p)
        lex_addr = index.lexicon.lookup(word)
        step.nodes[i] = <Lexeme*>lex_addr


cdef void free_sent(Sentence* s):
    cdef size_t i, j
    for i in range(s.n):
        free_step(&s.lattice[i])
    free(s.lattice)
    free(s.tokens)


cdef void free_step(Step* s):
    free(s.nodes)
    free(s.probs)


cdef class Input:
    def __init__(self, list lattice, list parse, prior=1.0, turn_id=None, wer=0):
        # Pad lattice with start and end tokens
        lattice.insert(0, [(1.0, '<start>')])
        parse.insert(0, (0, None, None, None, False, False))
        lattice.append([(1.0, '<end>')])
        parse.append((0, 'EOL', None, None, False, False))
        self.c_sent = init_sent(lattice, parse)
        self.wer = wer
        self.turn_id = turn_id
        self.prior = prior

    def __dealloc__(self):
        free_sent(self.c_sent)

    @classmethod
    def from_tokens(cls, tokens, turn_id=None, wer=0, prior=1.0):
        """
        Create sentence from a flat list of unambiguous tokens, instead of a lattice.
        Tokens should be a list of (word, tag, head, label, sent_id, is_edit)
        tuples
        """
        lattice = []
        parse = []
        for word, tag, head, label, sent_id, is_edit in tokens:
            lattice.append([(1.0, word)])
            parse.append((0, tag, head, label, sent_id, is_edit))
        return cls(lattice, parse, turn_id=turn_id, wer=wer, prior=prior)

    @classmethod
    def from_pos(cls, pos_str, prior=1.0):
        assert pos_str
        tokens = []
        for token_str in pos_str.split():
            word, pos = token_str.rsplit('/', 1)
            tokens.append((word, pos, None, None, None, None))
        return cls.from_tokens(tokens, prior=prior)

    @classmethod
    def from_conll(cls, conll_str):
        tokens = []
        turn_id = None
        for line in conll_str.split('\n'):
            fields = line.split()
            word_id = int(fields[0]) - 1
            word = fields[1]
            pos = fields[3]
            assert pos.strip()
            feats = fields[5].split('|')
            is_edit = len(feats) >= 3 and feats[2] == '1'
            #if len(feats) >= 5 and feats[4] == 'RM':
            #    is_edit = True
            is_fill = len(feats) >= 2 and feats[1] in ('D', 'E', 'F', 'A') 
            is_ns = len(feats) >= 4 and feats[3] == 'N_S'
            is_ns = False
            if '.' in feats[0]:
                turn_id, sent_id = feats[0].split('.')
            else:
                turn_id = feats[0]
                sent_id = '0'
            head = int(fields[6])
            label = fields[7]
            if is_edit:
                label = 'erased'
            elif is_fill:
                label = 'filler%s' % feats[1]
            elif is_ns:
                label = 'fillerNS'
            tokens.append((word, pos, head, label, int(sent_id),
                          is_edit or is_fill or is_ns))
        return cls.from_tokens(tokens, turn_id=turn_id)

    def segment(self):
        cdef size_t i
        cdef size_t root = encode_label('ROOT')
        cdef size_t cc = encode_label('cc')
        cdef size_t conj = encode_label('conj')
        cdef size_t nsubj = encode_label('nsubj')
        cdef Sentence* sent = self.c_sent
        cdef Token* token
        has_subj = set()
        conjunctions = []
        conjuncts = []
        roots = set()
        for i in range(1, self.c_sent.n - 1):
            token = &sent.tokens[i]
            if token.label == nsubj:
                has_subj.add(token.head)
            elif token.label == root:
                roots.add(i)
            elif token.label == cc:
                conjunctions.append(i)
            elif token.label == conj:
                conjuncts.append(i)
        for c in conjuncts:
            if c in has_subj and sent.tokens[c].head in roots:
                roots.add(c)
        left_edges = []
        for root in roots:
            left_edges.append(sent.tokens[root].left_edge)
        left_edges.sort()
        left_edges.reverse()
        segments = []
        last_left = sent.n - 1
        for edge in left_edges:
            if edge != 1 and sent.tokens[edge - 1].label == cc:
                edge -= 1
            segments.append((edge, last_left))
            last_left = edge
        if left_edges and edge >= 2:
            segments[-1] = (1, segments[-1][1])
        segments.sort()
        for i, (start, end) in enumerate(segments):
            for j in range(start, end):
                self.c_sent.tokens[j].sent_id = i


    property tokens:
        def __get__(self):
            Token = namedtuple('Token', 'id word tag head label sent_id is_edit')
            for i in range(1, self.c_sent.n - 1):
                word = index.lexicon.get_str(<size_t>self.c_sent.tokens[i].word)
                tag = decode_pos(self.c_sent.tokens[i].tag)
                head = self.c_sent.tokens[i].head
                label = decode_label(self.c_sent.tokens[i].label)
                is_edit = self.c_sent.tokens[i].is_edit
                sent_id = self.c_sent.tokens[i].sent_id
                yield Token(i, word, tag, head, label, sent_id, is_edit)

    property turn_id:
        def __get__(self):
            return self.turn_id

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

    property score:
        def __get__(self):
            return self.c_sent.score

    def to_conll(self):
        lines = []
        for i in range(1, self.length - 1):
            lines.append(conll_line_from_token(i, &self.c_sent.tokens[i],
                         self.c_sent.lattice))
        return '\n'.join(lines)


cdef object conll_line_from_token(size_t i, Token* a, Step* lattice):
    cdef bytes word = index.lexicon.get_str(<size_t>a.word)
    if not word:
        word = b'-OOV-'
    label = decode_label(a.label)
    fill_tag = label[-1] if label.startswith('filler') else '-'
    feats = '0.%d|%s|%d|-' % (a.sent_id, fill_tag, label == 'erased')
    cdef bytes tag = index.hashes.decode_pos(a.tag)
    assert tag.strip()
    return '\t'.join((str(i), word, '_', tag, tag, feats, 
                     str(a.head), label, '_', '_'))
