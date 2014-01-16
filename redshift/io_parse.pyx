from libc.stdlib cimport malloc, calloc, free
from libc.string cimport strcpy, memcpy
import index.hashes
cimport index.hashes


cdef Sentence* make_sentence(size_t id_, size_t length, py_words, py_tags,
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
    s.parens = <int*>calloc(size, sizeof(int))
    s.quotes = <int*>calloc(size, sizeof(int))
    s.non_alpha = <bint*>calloc(size, sizeof(bint))
    s.oft_upper = <bint*>calloc(size, sizeof(bint))
    s.oft_title = <bint*>calloc(size, sizeof(bint))

    cdef index.hashes.ClusterIndex brown_idx = index.hashes.get_clusters()
    cdef dict case_dict = index.hashes.get_case_stats()
    mask_value = index.hashes.encode_word('<MASKED>')
    cdef int paren_cnt = 0
    cdef int quote_cnt = 0
    types = set()
    for i in range(length):
        s.owords[i] = index.hashes.encode_word(py_words[i])
        if '-' in py_words[i] and py_words[i][0] != '-':
            word = '!HYPHEN'
        elif py_words[i].isdigit() and len(py_words[i]) == 4:
            word = '!YEAR'
        elif py_words[i][0].isdigit():
            word = '!DIGITS'
        else:
            word = py_words[i].lower()
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
        if thresh != 0 and index.hashes.get_freq(py_words[i]) <= thresh:
            s.words[i] = mask_value
        # Use POS tag to semi-smartly get ' disambiguation
        if py_tags[i] == "``":
            quote_cnt += 1
        elif py_tags[i] == "''":
            quote_cnt -= 1
        elif py_words[i] == "(" or py_words[i] == "[" or py_words[i] == "{":
            paren_cnt += 1
        elif py_words[i] == ")" or py_words[i] == "]" or py_words[i] == "}":
            paren_cnt -= 1
        s.suffix[i] = index.hashes.encode_word(py_words[i][-3:])
        s.prefix[i] = index.hashes.encode_word(py_words[i][0])
        s.parens[i] = paren_cnt
        s.quotes[i] = quote_cnt
    s.words[0] = 0
    s.pos[0] = 0
    return s


cdef int add_parse(Sentence* sent, list word_ids, list heads, list labels, edits) except -1:
    cdef size_t segment = 0
    for i in range(sent.length):
        sent.parse.heads[i] = <size_t>heads[i]
        sent.parse.labels[i] = index.hashes.encode_label(labels[i])
        if i >= 1 and word_ids[i] is not None and word_ids[i] >= word_ids[i - 1]:
            segment += 1
        sent.parse.sbd[i] = segment
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
    free(s.parens)
    free(s.quotes)
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


def read_conll(conll_str, moves=None, vocab_thresh=0, unlabelled=False):
    cdef:
        size_t i
        object words, tags, heads, labels, token_str, word, pos, head, label
        Sentences sentences
    sent_strs = conll_str.strip().split('\n\n')
    sentences = Sentences(max_length=len(sent_strs), vocab_thresh=vocab_thresh)
    first_sent = sent_strs[0]
    cdef size_t id_
    for id_, sent_str in enumerate(sent_strs):
        if not sent_str.split():
            continue
        words = ['<start>']
        tags = ['OOB']
        heads = [0]
        labels = ['ERR']
        ids = [None]
        token_strs = sent_str.split('\n')
        edits = [False]
        for tok_id, token_str in enumerate(token_strs):
            pieces = token_str.split()
            if len(pieces) == 10:
                word_id = int(pieces[0])
                word = pieces[1]
                pos = pieces[3]
                pos2 = pieces[4]
                feats = pieces[5].split('|')
                head = pieces[6]
                label = pieces[7]
                head = int(head) - 1
                if feats and feats[2] == '1':
                    is_edit = True
                else:
                    is_edit = False
                #if feats and feats[1] == 'D':
                #    label = 'discourse'
            else:   
                if len(pieces) == 5:
                    pieces.pop(0)
                try:
                    word, pos, head, label = pieces
                except:
                    print repr(token_str)
                    raise
                head = int(head)
                is_edit = False
                word_id = tok_id
            if unlabelled and label not in ['ROOT', 'P', 'conj', 'cc', 'erased',
					    'discourse', 'interregnum']:
                label = 'ERR'
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
            ids.append(word_id)
        ids.append(None)
        words.append('<root>')
        tags.append('ROOT')
        heads.append(0)
        labels.append('ERR')
        edits.append(False)
        sent = make_sentence(id_, len(ids), words, tags, vocab_thresh)
        add_parse(sent, ids, heads, labels, edits)
        sentences.add(sent, words, tags)
    if moves is not None and moves.strip():
        sentences.add_moves(moves)
    return sentences

    
def read_pos(file_str, vocab_thresh=0, sep='/'):
    cdef:
        size_t i
        object token_str, word, pos, words, tags
        Sentences sentences

    sent_strs = file_str.strip().split('\n')
    sentences = Sentences(max_length=len(sent_strs), vocab_thresh=vocab_thresh)
    cdef size_t w_id = 0
    for i, sent_str in enumerate(sent_strs):
        if not sent_str.strip():
            continue
        words = ['<start>']
        tags = ['OOB']
        ids = [None]
        for token_str in sent_str.split():
            try:
                word, pos = token_str.rsplit(sep, 1)
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
        ids.append(None)
        sent = make_sentence(i, len(ids), words, tags, vocab_thresh)
        sentences.add(sent, words, tags)
    return sentences


cdef class Sentences:
    def __cinit__(self, size_t max_length=100000, vocab_thresh=0):
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
        return tags, list(sorted(left_labels)), list(sorted(right_labels))

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
                if s.parse.heads[j] == s.length - 1:
                    head = -1
                else:
                    head = <int>(s.parse.heads[j]) - (j - w_id)
                fields = (w_id, py_words[j], pos_idx[s.pos[j]], head,
                          label_idx.get(s.parse.labels[j], 'ERR'))
                out_file.write(u'%d\t%s\t%s\t%s\t%s\n' % fields)
                w_id += 1
                if s.parse.sbd[j] or j == (s.length - 2):
                    out_file.write(u'\n')
                    w_id = 0

    def write_tags(self, out_file):
        cdef Sentence* s
        cdef size_t i, j
        pos_idx = index.hashes.reverse_pos_index()
        for i in range(self.length):
            s = self.s[i]
            py_words, py_pos = self.strings[i]
            w_id = 0
            for j in range(1, s.length - 1):
                fields = (py_words[j], pos_idx[s.pos[j]])
                out_file.write(u'%s/%s ' % fields)
            out_file.write(u'\n')

    property scores:
        def __get__(self):
            scores = []
            cdef object score
            for i in range(self.length):
                score = self.s[i].parse.score
                scores.append(score)
            return scores
      
    property length:
        def __get__(self): return self.length

def eval_tags(Sentences test, Sentences gold):
    c = 0
    ac = 0
    n = 0
    assert test.length == gold.length
    for i in range(test.length):
        assert test.s[i].length == gold.s[i].length
        for w in range(1, test.s[i].length - 1):
            c += test.s[i].pos[w] == gold.s[i].pos[w]
            ac += (test.s[i].pos[w] == gold.s[i].pos[w] or test.s[i].alt_pos[w] == gold.s[i].pos[w])
            n += 1

    print (float(ac)/n) * 100, ac, n
    return (float(c)/n) * 100, c, n
