from libc.stdlib cimport malloc, calloc, free

cimport index.vocab
cimport index.hashes
cimport index.vocab


cdef Sentence* init_sent(list words_cn, object tags, object parse):
    cdef Sentence* s = <Sentence*>malloc(sizeof(Sentence))
    s.n = len(words_cn)
    s.steps = <Step*>calloc(s.n, sizeof(Step))
    for i in range(len(words_cn)):
        init_step(words_cn[i], &s.steps[i])
    # We used to have to do this to ensure that the zero-position evaluated
    # to 0, for the feature calculation (0 is a special value indicating absent)
    # There's probably a better way, but for now...
    cdef Word* w
    for i in range(s.steps[0].n):
        s.steps[0].nodes[i].orig = 0
    # For the output
    s.answer = <AnswerToken*>calloc(s.n, sizeof(AnswerToken))
    if tags is not None:
        fill_tags(tags, s.answer)
    if parse is not None:
        fill_answer(0.0, parse, s.answer)
    return s


cdef int init_step(list cn_step, Step* step):
    step.n = len(cn_step)
    step.nodes = <Word**>calloc(step.n, sizeof(Word*))
    step.probs = <double*>calloc(step.n, sizeof(double))
    for i, (p, word) in enumerate(cn_step):
        step.probs[i] = p
        step.nodes[i] = index.vocab.lookup(word)


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


cdef class PySentence:
    def __init__(self, size_t id_, tokens, tags=None, parse=None):
        self.id = id_
        # Pad tokens with start 
        tokens.insert(0, ('<start>',))
        parse.insert(0, (0, 'ERR', False, False))
        # and end sentinels
        tokens.append(('<root>', 'ROOT'))
        parse.append((0, 'ERR', False, True))
        # Make the C representations the parser uses
        self.c_sent = init_sent(tokens, tags, parse)
        # Hold onto the Python tokens, but not the Python parse representation,
        # because we might want to display the analysis the parser gives us back,
        # so we read it off the C struct, which the parser manipulates.
        self.tokens = tokens
        self.length = len(tokens)

    def __dealloc__(self):
        free_sent(self.c_sent)

    @classmethod
    def from_conll(cls, size_t id_, object sent_str):
        pass

    @classmethod
    def from_pos(cls, id_, object sent_str):
        pass

    property length:
        def __get__(self): return self.length

    property tokens:
        def __get__(self): return self.tokens

    def to_conll(self):
        return ''


def get_labels(sents):
    tags = set()
    left_labels = set()
    right_labels = set()
    cdef size_t i
    cdef PySentence sent
    for i, sent in enumerate(sents):
        for j in range(sent.n):
            tags.add(sent.c_sent.answer[j].tag)
            if sent.c_sent.answer[j].head > j:
                left_labels.add(sent.c_sent.answer[j].label)
            else:
                right_labels.add(sent.c_sent.answer[j].label)
    return tags, list(sorted(left_labels)), list(sorted(right_labels))
