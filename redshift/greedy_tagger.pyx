# cython: profile=True
from features.extractor cimport Extractor
from learn.perceptron cimport Perceptron
import index.hashes
cimport index.hashes
from index.hashes import decode_pos
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


def get_tagset(sents):
    # Dict instead of set so json serialisable
    tags = {}
    freqdist = {}
    cdef size_t word
    cdef size_t tag
    cdef Input py_sent
    cdef Sentence* sent
    for py_sent in sents:
        sent = py_sent.c_sent
        for i in range(1, sent.n - 1):
            tag = sent.tokens[i].tag
            tags[tag] = 1
            word = sent.tokens[i].word.orig
            freqdist.setdefault(word, {}).setdefault(tag, 0)
            freqdist[word][tag] += 1
    return tags, freqdist
 

def train(train_str, model_dir, features='basic', nr_iter=10,
          feat_thresh=10):
    cdef Input sent
    cdef size_t i
    if path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    sents = [Input.from_pos(s) for s in train_str.strip().split('\n') if s.strip()]
    tags, tag_freqs = get_tagset(sents)
    Config.write(model_dir, 'tagger', features=features,
                 feat_thresh=feat_thresh, tags=tags, freqdist=tag_freqs)
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
        self.extractor = Extractor(basic + clusters + orth + case, [], bag_of_words=[])
        self._guessed = 0
        self._features = <uint64_t*>calloc(self.extractor.nr_feat, sizeof(uint64_t))
        self._context = <size_t*>calloc(CONTEXT_SIZE, sizeof(size_t))
        self.tagdict.set_empty_key(0)
        if path.exists(path.join(model_dir, 'pos')):
            index.hashes.load_pos_idx(path.join(model_dir, 'pos'))
        nr_tag = index.hashes.get_nr_pos()
        self.guide = Perceptron(nr_tag, path.join(model_dir, 'tagger.gz'))
        if path.exists(path.join(model_dir, 'tagger.gz')):
            self.guide.load(path.join(model_dir, 'tagger.gz'),
                            thresh=self.cfg.feat_thresh)
        self._make_tagdict(self.cfg.freqdist)

    def _make_tagdict(self, dict tag_freqs):
        cdef uint64_t word
        cdef size_t tag
        cdef size_t mode
        cdef dict freqs
        cdef object word_str
        cdef object t
        for word_str, freqs in tag_freqs.items():
            word = int(word_str)
            total = sum([int(f) for f in freqs.values()])
            if total >= 100:
                mode, tag_str = max([(int(f), t) for t, f in freqs.items()])
                tag = int(tag_str)
                if float(mode) / total >= 0.99:
                    assert tag != 0
                    self.tagdict[word] = tag

    cdef int tag_word(self, Token* tokens, size_t i, Step* lattice, size_t n) except -1:
        self._guessed = 0
        self._cache_hit = False
        if i >= n:
            return tokens[0].tag
        if i == 0:
            return tokens[0].tag
        cdef size_t lookup = self.tagdict[tokens[i].word.orig]
        if lookup != 0:
            tokens[i].tag = lookup
            self.guide.n_corr += 1
            self.guide.total += 1
            return lookup
        cdef double* scores
        fill_slots(&self.slots, tokens, i, lattice, n)
        scores = self.guide.cache.lookup(sizeof(Slots), &self.slots, &self._cache_hit)
        if not self._cache_hit:
            fill_context(self._context, self.slots.pp_tag, self.slots.p_tag,
                         self.slots.pp_word, self.slots.p_word, self.slots.n0,
                         self.slots.n1, self.slots.n2, self.slots.n3)
            self.extractor.extract(self._features, self._context)
            self.guide.fill_scores(self._features, scores)
        cdef size_t clas
        cdef size_t best = 1
        for clas in range(2, self.guide.nr_class):
            if scores[clas] > scores[best]:
                best = clas
        self._guessed = best
        tokens[i].tag = best

    cdef int tell_gold(self, size_t gold) except -1:
        if self._guessed == 0 or gold == 0:
            return 0
        if self._guessed != gold and self._cache_hit:
            fill_context(self._context, self.slots.pp_tag, self.slots.p_tag,
                         self.slots.pp_word, self.slots.p_word, self.slots.n0,
                         self.slots.n1, self.slots.n2, self.slots.n3)
            self.extractor.extract(self._features, self._context)
        self.guide.update(self._guessed, gold, self._features, 1.0)
 

cdef int fill_slots(Slots* slots, Token* state, size_t i, Step* lattice, size_t n):
    slots.pp_tag = state[i-1].tag if i >= 1 else 0
    slots.p_tag = state[i-2].tag if i >= 2 else 0
    slots.pp_word = state[i - 2].word if i >= 2 else NULL
    slots.p_word = state[i - 1].word if i >= 1 else NULL
    slots.n0 = state[i].word
    slots.n1 = lattice[i+1].nodes[0] if (i+1) < n else NULL
    slots.n2 = lattice[i+2].nodes[0] if (i+2) < n else NULL
    slots.n3 = lattice[i+3].nodes[0] if (i+3) < n else NULL


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
