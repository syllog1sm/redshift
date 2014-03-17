from libc.stdlib cimport malloc, calloc, free

cimport index.vocab
cimport index.hashes
import index.vocab

cdef Sentence* init_sent(list words_cn, object tags, object parse) except NULL:
    cdef Sentence* s = <Sentence*>malloc(sizeof(Sentence))
    s.n = len(words_cn)
    s.steps = <Step*>calloc(s.n, sizeof(Step))
    s.answer = <AnswerToken*>calloc(s.n, sizeof(AnswerToken))
    for i in range(len(words_cn)):
        init_step(words_cn[i], &s.steps[i])
        # TODO: Support passing in the answer for a lattice
        s.answer[i].word = s.steps[i].nodes[0]
    # We used to have to do this to ensure that the zero-position evaluated
    # to 0, for the feature calculation (0 is a special value indicating absent)
    # There's probably a better way, but for now...
    cdef Word* w
    for i in range(s.steps[0].n):
        s.steps[0].nodes[i].orig = 0
    # For the output
    if tags is not None:
        fill_tags(tags, s.answer)
    if parse is not None:
        fill_answer(0.0, parse, s.answer)
    return s


cdef int init_step(list cn_step, Step* step) except -1:
    step.n = len(cn_step)
    step.nodes = <Word**>calloc(step.n, sizeof(Word*))
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


cdef int fill_tags(list py_tags, AnswerToken* tokens):
    cdef size_t i
    cdef bytes tag
    for i, tag in enumerate(py_tags):
        tokens[i].tag = index.hashes.encode_pos(tag)
    # We used to have to do this to ensure that the zero-position evaluated
    # to 0, for the feature calculation (0 is a special value indicating absent)
    # There's probably a better way, but for now...
    tokens[0].tag = 0
 

cdef int fill_answer(double score, list py_parse, AnswerToken* tokens):
    cdef size_t head
    cdef bytes py_label
    cdef bint is_break
    cdef bint is_edit
    cdef size_t i
    for i, (head, py_label, is_break, is_edit) in enumerate(py_parse):
        tokens[i].head = head
        tokens[i].label = index.hashes.encode_label(py_label)
        tokens[i].is_break = is_break
        tokens[i].is_edit = is_edit


cdef class Input:
    def __init__(self, list words_cn):
        index.vocab.load_vocab()
        self.c_sent = init_sent(words_cn, None, None)

    def __dealloc__(self):
        # TODO: Fix memory
        pass
        #free_sent(self.c_sent)

    property length:
        def __get__(self):
            return self.c_sent.n

    property words:
        def __get__(self):
            return [index.vocab.get_str(<size_t>self.c_sent.answer[i].word)
                    for i in range(self.c_sent.n)]

    property tags:
        def __get__(self):
            return [self.c_sent.answer[i].tag for i in range(self.c_sent.n)]

    property heads:
        def __get__(self):
            return [self.c_sent.answer[i].head for i in range(self.c_sent.n)]

    property labels:
        def __get__(self):
            return [self.c_sent.answer[i].label for i in range(self.c_sent.n)]

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
          .word:str, .pos:str, .head:int, .label:str, is_edit:bool, is_break:bool
        - Sequences, of the form (str, str, int, str, bool, bool)
        """
        if hasattr(tokens[0], 'encode'):
            tokens = [Token.from_str(i, t) for i, t in enumerate(tokens)]
        elif hasattr(tokens[0], 'word') and hasattr(tokens[0], 'is_break'):
            tokens = [Token.from_obj(t) for t in tokens]
        else:
            raise StandardError("Could not guess how to parse input: %s" % tokens[0])
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
            head = int(fields[6]) - 1
            label = fields[7]
            tokens.append(Token(word_id, pos, head, label, is_edit, is_break))
        return cls.from_tokens(tokens)


cdef class Token:
    def __init__(self, i, word, pos=None, head=None, label=None,
                 is_edit=False, is_break=False):
        self.i = i
        self.c_word = <Word*>index.vocab.lookup(word)
        self.c_parse = <AnswerToken*>malloc(sizeof(AnswerToken))
        if pos is not None:
            self.c_parse.tag = index.hashes.encode_pos(pos)
        if head is not None:
            self.c_parse.head = head
        if label is not None:
            self.c_parse.label = label
        self.c_parse.is_edit = is_edit
        self.c_parse.is_break = is_break

    def __dealloc__(self):
        free(self.c_parse)

    def __repr__(self):
        return 'Token(%s)' % (index.vocab.get_str(<size_t>self.c_word))

    @classmethod
    def from_str(cls, i, token_str):
        fields = token_str.split('/')
        if len(fields) < 2:
            fields.append(None)
        word, pos = fields
        return cls(i, word, pos=pos)

    @classmethod
    def from_obj(cls, t):
        feats = ['0', '0', '1' if t.is_edit else '0', '1' if t.is_break else '0']
        return cls(t.i, t.word, t.pos, feats, t.head, t.label)


def get_labels(sents):
    tags = set()
    left_labels = set()
    right_labels = set()
    cdef size_t i
    cdef Input sent
    for i, sent in enumerate(sents):
        for j in range(sent.n):
            tags.add(sent.c_sent.answer[j].tag)
            if sent.c_sent.answer[j].head > j:
                left_labels.add(sent.c_sent.answer[j].label)
            else:
                right_labels.add(sent.c_sent.answer[j].label)
    return tags, list(sorted(left_labels)), list(sorted(right_labels))
