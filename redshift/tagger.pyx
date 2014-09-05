# cython: profile=True
from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
import index.hashes
cimport index.hashes
from index.lexicon cimport get_str
from .util import Config

from redshift.sentence cimport Input, Sentence, Step
from index.lexicon cimport Lexeme

from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport uint64_t, int64_t
from libcpp.queue cimport priority_queue
from libcpp.utility cimport pair

cimport cython
import os
from os import path
import random
import shutil


def train(train_str, model_dir, beam_width=4, features='basic', nr_iter=10,
          feat_thresh=10):
    cdef Input sent
    cdef size_t i
    if path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    sents = [Input.from_pos(s) for s in train_str.strip().split('\n') if s.strip()]
    # Dict instead of set so json serialisable
    tags = {}
    for sent in sents:
        for i in range(sent.c_sent.n):
            tags[sent.c_sent.tokens[i].tag] = 1
    Config.write(model_dir, 'tagger', beam_width=beam_width, features=features,
                 feat_thresh=feat_thresh, tags=tags)
    tagger = Tagger(model_dir)
    indices = list(range(len(sents)))
    for n in range(nr_iter):
        for i in indices:
            sent = sents[i]
            tagger.train_sent(sent)
        tagger.guide.end_train_iter(n, feat_thresh)
        random.shuffle(indices)
    tagger.guide.end_training(path.join(model_dir, 'tagger.gz'))
    index.hashes.save_pos_idx(path.join(model_dir, 'pos'))
    return tagger


cdef class Tagger:
    def __cinit__(self, model_dir):
        self.cfg = Config.read(model_dir, 'tagger')
        self.extractor = Extractor(basic + clusters + case + orth, [],
                                   bag_of_words=[])
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))

        self.beam_width = self.cfg.beam_width

        if path.exists(path.join(model_dir, 'pos')):
            index.hashes.load_pos_idx(path.join(model_dir, 'pos'))
        nr_tag = index.hashes.get_nr_pos()
        self.guide = Perceptron(nr_tag, path.join(model_dir, 'tagger.gz'))
        if path.exists(path.join(model_dir, 'tagger.gz')):
            self.guide.load(path.join(model_dir, 'tagger.gz'),
                            thresh=self.cfg.feat_thresh)
        self._beam_scores = <double**>malloc(sizeof(double*) * self.beam_width)
        for i in range(self.beam_width):
            self._beam_scores[i] = <double*>calloc(nr_tag, sizeof(double))

    cdef int tag_word(self, Token* state, size_t i, Step* lattice, size_t n):
        if i >= n:
            return 0
        if i == 0:
            state[0].tag = 0
            return 0
        cdef size_t prev_tag = state[i-1].tag if i >= 1 else 0
        cdef size_t prevprev_tag = state[i-2].tag if i >= 2 else 0
        cdef Lexeme* prev_word = state[i - 1].word if i >= 1 else NULL
        cdef Lexeme* prevprev_word = state[i - 2].word if i >= 2 else NULL
        cdef Lexeme* n0 = lattice[i].nodes[0]
        cdef Lexeme* n1 = lattice[i+1].nodes[0] if (i+1) < n else NULL
        cdef Lexeme* n2 = lattice[i+2].nodes[0] if (i+2) < n else NULL
        cdef Lexeme* n3 = lattice[i+3].nodes[0] if (i+3) < n else NULL
        fill_context(self._context, prevprev_tag, prev_tag, prevprev_word, prev_word,
                     n0, n1, n2, n3)
        self.extractor.extract(self._features, self._context)
        self.guide.fill_scores(self._features, self.guide.scores)
        cdef double mode = self.guide.scores[0]
        state[i].tag = 0
        for clas in range(self.guide.nr_class):
            if self.guide.scores[clas] > mode:
                state[i].tag = clas
                mode = self.guide.scores[clas]


    cpdef int tag(self, Input py_sent) except -1:
        cdef Sentence* sent = py_sent.c_sent
        cdef TaggerBeam beam = TaggerBeam(self.beam_width, sent.n, self.guide.nr_class)
        cdef size_t p_idx
        cdef TagState* s
        for i in range(sent.n - 1):
            # Extend beam
            for j in range(beam.bsize):
                # At this point, beam.clas is the _last_ prediction, not the
                # prediction for this instance
                self._predict(i, beam.parents[j], sent, self._beam_scores[j])
            beam.extend_states(self._beam_scores)
        s = <TagState*>beam.beam[0]
        fill_hist(sent.tokens, s, sent.n - 1)

    cdef int train_sent(self, Input py_sent) except -1:
        cdef size_t  i, j 
        cdef Sentence* sent = py_sent.c_sent
        cdef size_t nr_class = self.guide.nr_class
        cdef double* scores = self.guide.scores
        cdef TaggerBeam beam = TaggerBeam(self.beam_width, sent.n, nr_class)
        cdef TagState* gold = extend_state(NULL, 0, NULL, 0)
        cdef MaxViolnUpd updater = MaxViolnUpd(nr_class)
        for i in range(sent.n - 1):
            # Extend gold
            self._predict(i, gold, sent, scores)
            gold = extend_state(gold, sent.tokens[i].tag, scores, nr_class)
            # Extend beam
            for j in range(beam.bsize):
                # At this point, beam.clas is the _last_ prediction, not the
                # prediction for this instance
                self._predict(i, beam.parents[j], sent, self._beam_scores[j])
            beam.extend_states(self._beam_scores)
            updater.compare(beam.beam[0], gold, i)
            self.guide.n_corr += (gold.clas == beam.beam[0].clas)
            self.guide.total += 1
        if updater.delta != -1:
            counts = updater.count_feats(self._features, self._context, sent,
                                         self.extractor)
            self.guide.batch_update(counts)
        cdef TagState* prev
        while gold != NULL:
            prev = gold.prev
            free(gold)
            gold = prev

    cdef int _predict(self, size_t i, TagState* s, Sentence* sent, double* scores):
        fill_context(self._context, get_p(s), s.clas, get_token(sent, i, -2),
                     get_token(sent, i, -1), get_token(sent, i, 0),
                     get_token(sent, i, 1), get_token(sent, i, 2),
                     get_token(sent, i, 3))
        self.extractor.extract(self._features, self._context)
        self.guide.fill_scores(self._features, scores)
        cdef double mode = scores[0]
        best = 0
        for clas in range(self.guide.nr_class):
            if scores[clas] > mode:
                best = clas
                mode = scores[clas]
        # TODO Isn't this missing the assignment?




cdef class MaxViolnUpd:
    cdef TagState* pred
    cdef TagState* gold
    cdef Sentence* sent
    cdef double delta
    cdef int length
    cdef size_t nr_class
    cdef size_t tmp
    def __cinit__(self, size_t nr_class):
        self.delta = -1
        self.length = -1
        self.nr_class = nr_class

    cdef int compare(self, TagState* pred, TagState* gold, size_t i):
        delta = pred.score - gold.score
        if delta > self.delta:
            self.delta = delta
            self.pred = pred
            self.gold = gold
            self.length = i 

    cdef dict count_feats(self, uint64_t* feats, size_t* context, Sentence* sent,
                          Extractor extractor):
        if self.length == -1:
            return {}
        cdef TagState* g = self.gold
        cdef TagState* p = self.pred
        cdef int i = self.length
        cdef dict counts = {}
        for clas in range(self.nr_class):
            counts[clas] = {} 
        cdef size_t gclas, gprev, gprevprev
        cdef size_t pclas, pprev, prevprev
        while g != NULL and p != NULL and i >= 0:
            gclas = g.clas
            gprev = get_p(g)
            gprevprev = get_pp(g)
            pclas = p.clas
            pprev = get_p(p)
            pprevprev = get_pp(p)
            if gclas == pclas and pprev == gprev and gprevprev == pprevprev:
                g = g.prev
                p = p.prev
                i -= 1
                continue
            fill_context(context, gprevprev, gprev, get_token(sent, i, -2),
                         get_token(sent, i, -1), get_token(sent, i, 0), get_token(sent, i, 1),
                         get_token(sent, i, 2), get_token(sent, i, 3))
            extractor.extract(feats, context)
            self._inc_feats(counts[gclas], feats, 1.0)
            context[P1p] = pprev
            context[P2p] = pprevprev
            #fill_context(context, sent, pprev, pprevprev, i)
            extractor.extract(feats, context)
            self._inc_feats(counts[p.clas], feats, -1.0)
            assert sent.tokens[i].word.norm == context[N0w]
            g = g.prev
            p = p.prev
            i -= 1
        return counts

    cdef int _inc_feats(self, dict counts, uint64_t* feats, double inc) except -1:
        cdef size_t f = 0
        while feats[f] != 0:
            if feats[f] not in counts:
                counts[feats[f]] = 0
            counts[feats[f]] += inc
            f += 1

cdef Lexeme* get_token(Sentence* sent, size_t i, int offset):
    cdef int index = <int>i + offset
    if index < 1:
        return NULL
    if index >= sent.n:
        return NULL
    else:
        return sent.tokens[index].word


cdef enum:
    P1p
    P2p

    N0w
    N0c
    N0c6
    N0c4
    N0pre
    N0suff
    N0title
    N0upper
    N0alpha

    N1w
    N1c
    N1c6
    N1c4
    N1pre
    N1suff
    N1title
    N1upper
    N1alpha

    N2w
    N2c
    N2c6
    N2c4
    N2pre
    N2suff
    N2title
    N2upper
    N2alpha

    N3w
    N3c
    N3c6
    N3c4
    N3pre
    N3suff
    N3title
    N3upper
    N3alpha

    P1w
    P1c
    P1c6
    P1c4
    P1pre
    P1suff
    P1title
    P1upper
    P1alpha

    P2w
    P2c
    P2c6
    P2c4
    P2pre
    P2suff
    P2title
    P2upper
    P2alpha

    CONTEXT_SIZE


basic = (
    (N0w,),
    (N1w,),
    (P1w,),
    (P2w,),
    (P1p,),
    (P2p,),
    (P1p, P2p),
    (P1p, N0w),
    (N0suff,),
    (N1suff,),
    (P1suff,),
    (N2w,),
    (N3w,),
    (P1p,),
)

case = (
    (N0title,),
    (N0upper,),
    (N0alpha,),
    (N0title, N0suff),
    (N0title, N0upper, N0alpha),
    (P1title,),
    (P1upper,),
    (P1alpha,),
    (N1title,),
    (N1upper,),
    (N1alpha,),
    (P1title, N0title, N1title),
    (P1p, N0title,),
    (P1p, N0upper,),
    (P1p, N0alpha,),
    (P1title, N0w),
    (P1upper, N0w),
    (P1title, N0w, N1title),
    (N0title, N0upper, N0c),
)

orth = (
    (N0pre,),
    (N1pre,),
    (P1pre,),
)

clusters = (
    (N0c,),
    (N0c4,),
    (N0c6,),
    (P1c,),
    (P1c4,),
    (P1c6,),
    (N1c,),
    (N1c4,),
    (N1c6,),
    (N2c,),
    (P1c, N0w),
    (P1p, P1c6, N0w),
    (P1c6, N0w),
    (N0w, N1c),
    (N0w, N1c6),
    (N0w, N1c4),
    (P2c4, P1c4, N0w)
)


cdef inline void fill_token(size_t* context, size_t i, Lexeme* word):
    if word is not NULL:
        context[i] = word.norm
        # We've read in the string little-endian, so now we can take & (2**n)-1
        # to get the first n bits of the cluster.
        # e.g. s = "1110010101"
        # s = ''.join(reversed(s))
        # first_4_bits = int(s, 2)
        # print first_4_bits
        # 5
        # print "{0:b}".format(prefix).ljust(4, '0')
        # 1110
        # What we're doing here is picking a number where all bits are 1, e.g.
        # 15 is 1111, 63 is 111111 and doing bitwise AND, so getting all bits in
        # the source that are set to 1.
        context[i+1] = word.cluster
        context[i+2] = word.cluster & 63
        context[i+3] = word.cluster & 15
        context[i+4] = word.prefix
        context[i+5] = word.suffix
        context[i+6] = word.oft_title
        context[i+7] = word.oft_upper
        context[i+8] = word.non_alpha

cdef int fill_context(size_t* context,
                      size_t pptag, size_t ptag,
                      Lexeme* p2, Lexeme* p1,
                      Lexeme* n0,
                      Lexeme* n1, Lexeme* n2, Lexeme* n3):
    for j in range(CONTEXT_SIZE):
        context[j] = 0
    context[P1p] = ptag
    context[P2p] = pptag
    
    fill_token(context, N0w, n0)
    fill_token(context, N1w, n1)
    fill_token(context, N2w, n2)
    fill_token(context, N3w, n3)
    fill_token(context, P1w, p1)
    fill_token(context, P2w, p2)


cdef class TaggerBeam:
    def __cinit__(self, size_t k, size_t length, nr_tag=None):
        self.nr_class = nr_tag
        self.k = k
        self.t = 0
        self.bsize = 1
        self.is_full = self.bsize >= self.k
        self.seen_states = set()
        self.beam = <TagState**>malloc(k * sizeof(TagState*))
        self.parents = <TagState**>malloc(k * sizeof(TagState*))
        cdef size_t i
        for i in range(k):
            self.parents[i] = extend_state(NULL, 0, NULL, 0)
            self.seen_states.add(<size_t>self.parents[i])

    @cython.cdivision(True)
    cdef int extend_states(self, double** ext_scores) except -1:
        # Former states are now parents, beam will hold the extensions
        cdef size_t i, clas, move_id
        cdef double parent_score, score
        cdef double* scores
        cdef priority_queue[pair[double, size_t]] next_moves
        next_moves = priority_queue[pair[double, size_t]]()
        for i in range(self.bsize):
            scores = ext_scores[i]
            for clas in range(self.nr_class):
                score = self.parents[i].score + scores[clas]
                move_id = (i * self.nr_class) + clas
                next_moves.push(pair[double, size_t](score, move_id))
        cdef pair[double, size_t] data
        # Apply extensions for best continuations
        cdef TagState* s
        cdef TagState* prev
        cdef size_t addr
        cdef dense_hash_map[uint64_t, bint] seen_equivs = dense_hash_map[uint64_t, bint]()
        seen_equivs.set_empty_key(0)
        self.bsize = 0
        while self.bsize < self.k and not next_moves.empty():
            data = next_moves.top()
            i = data.second / self.nr_class
            clas = data.second % self.nr_class
            prev = self.parents[i]
            hashed = (clas * self.nr_class) + prev.clas
            if seen_equivs[hashed]:
                next_moves.pop()
                continue
            seen_equivs[hashed] = 1
            self.beam[self.bsize] = extend_state(prev, clas, ext_scores[i],
                                                 self.nr_class)
            addr = <size_t>self.beam[self.bsize]
            self.seen_states.add(addr)
            next_moves.pop()
            self.bsize += 1
        for i in range(self.bsize):
            self.parents[i] = self.beam[i]
        self.is_full = self.bsize >= self.k
        self.t += 1

    def __dealloc__(self):
        cdef TagState* s
        cdef size_t addr
        for addr in self.seen_states:
            s = <TagState*>addr
            free(s)
        free(self.parents)
        free(self.beam)


cdef TagState* extend_state(TagState* s, size_t clas, double* scores,
                            size_t nr_class):
    cdef double score
    ext = <TagState*>calloc(1, sizeof(TagState))
    ext.prev = s
    ext.clas = clas
    if s == NULL:
        ext.score = 0
        ext.length = 0
    else:
        ext.score = s.score + scores[clas]
        ext.length = s.length + 1
    return ext


cdef int fill_hist(Token* tokens, TagState* s, int t) except -1:
    while t >= 1 and s.prev != NULL:
        t -= 1
        tokens[t].tag = s.clas
        s = s.prev

cdef size_t get_p(TagState* s):
    if s.prev == NULL:
        return 0
    else:
        return s.prev.clas


cdef size_t get_pp(TagState* s):
    if s.prev == NULL:
        return 0
    elif s.prev.prev == NULL:
        return 0
    else:
        return s.prev.prev.clas
