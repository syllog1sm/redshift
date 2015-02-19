# cython: profile=True
"""
MALT-style dependency parser
"""
cimport cython
import random
import os.path
from os.path import join as pjoin
import shutil
import json

from libc.string cimport memcpy, memset

from cymem.cymem cimport Pool, Address
from thinc.typedefs cimport weight_t, class_t, feat_t, atom_t
from thinc.search cimport Beam, MaxViolation

from _state cimport *
from sentence cimport Input, Sentence, Token, Step

from tagger cimport Tagger
from util import Config

from thinc.features cimport Extractor
from thinc.features cimport Feature
from thinc.features cimport count_feats
import _parse_features
from _parse_features cimport *

import index.hashes
cimport index.hashes

from thinc.learner cimport LinearModel


include "compile_time_options.pxi"
IF TRANSITION_SYSTEM == 'arc_eager':
    from .arc_eager cimport *
ELSE:
    from .arc_hybrid cimport *


VOCAB_SIZE = 1e6
TAG_SET_SIZE = 50


DEBUG = False 
def set_debug(val):
    global DEBUG
    DEBUG = val


def train(train_str, model_dir, n_iter=15, beam_width=8,
          train_tagger=True, feat_set=u'basic', feat_thresh=0, seed=0,
          use_edit=False, use_break=False, use_filler=False):
    """Train a model from a CoNLL-formatted training string, creating a model in
    model_dir.

    Args:
        train_str (bytes): A CoNLL-formatted string, describing the training
            sentences. See Input.from_conll.
        model_dir (bytes): The path that the model will be saved in. Will be
            wiped if it exists, and re-created.
        n_iter (int): The number of iterations to train for. Default 15.
        beam_width (int): The number of candidates to maintain in the `beam'. Efficiency
            is usually slightly less than linear in the width of the beam (because)
            we do some caching). Accuracy typically plateaus between 16 and 32
            on English. If the model is more accurate, less beam width is necessary.
        train_tagger (bool): Whether to train a tagger alongside the parser. The
            parser is trained with tags predicted by the current tagger model,
            allowing it to see semi-realistic examples of predicted tags. This
            tends to be as good as jack-knife training the tagger, but with less
            complexity.
        feat_set (unicode): A string describing the features to be used. See the
            get_templates function for details about the various feature-set
            names. The string is searched for the presence of substrings, so there's
            some flexibility in how you can write the string. For instance, both
            clusters+bitags and clusters/bitags will add the "clusters" and "bitags"
            feature sets. The basic feature set is added by default.
        feat_thresh (int): Post-prune the feature set, removing features that occur
            N or fewer times. This is currently not recommended --- it performs
            poorly with the ADADELTA training.
        seed (int): Seed the shuffling of sentences between iterations. This is
            useful to conduct N trials of an experimental configuration, allowing
            significance testing. Variance is often 0.1-0.3% UAS, making this
            quite a powerful way to tell whether two runs are meaningfully
            different.
        use_edit (bint): Controls whether to use the Edit transition, for speech
            parsing.
        use_break (bint): Controls whether to use the Break transition, for
            speech parsing.
        """

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)
    cdef list sents = [Input.from_conll(s) for s in
                       train_str.strip().split('\n\n') if s.strip()]
    left_labels, right_labels, dfl_labels = get_labels(sents)
    Config.write(model_dir, 'config', beam_width=beam_width, features=feat_set,
                 feat_thresh=feat_thresh, seed=seed,
                 left_labels=left_labels, right_labels=right_labels,
                 dfl_labels=dfl_labels, use_break=use_break)
    Config.write(model_dir, 'tagger', beam_width=4, features='basic',
                 feat_thresh=5, tags={})
    parser = Parser(model_dir)
    indices = list(range(len(sents)))
    cdef Input py_sent
    for n in range(n_iter):
        for i in indices:
            py_sent = sents[i]
            parser.tagger.train_sent(py_sent)
            parser.train_sent(py_sent)
        acc = float(parser.guide.n_corr) / parser.guide.total
        print(parser.guide.end_train_iter(n, feat_thresh) + '\t' +
              parser.tagger.guide.end_train_iter(n, feat_thresh))
        random.shuffle(indices)
    parser.guide.end_training()
    parser.tagger.guide.end_training()
    parser.guide.dump(pjoin(model_dir, 'model'), freq_thresh=0)
    parser.tagger.guide.dump(pjoin(model_dir, 'tagger'))
    index.hashes.save_pos_idx(pjoin(model_dir, 'pos'))
    index.hashes.save_label_idx(pjoin(model_dir, 'labels'))
    return acc


def get_labels(sents):
    '''Get alphabetically-sorted lists of left, right and disfluency labels that
    occur in a sample of sentences. Used to determine the set of legal transitions
    from the training set.

    Args:
        sentences (list[Input]): A list of Input objects, usually the training set.

    Returns:
        labels (tuple[list, list, list]): Sorted lists of left, right and disfluency
            labels.
    '''
    left_labels = set()
    right_labels = set()
    dfl_labels = set()
    cdef Input sent
    for i, sent in enumerate(sents):
        for j in range(sent.length):
            if sent.c_sent.tokens[j].is_edit:
                dfl_labels.add(sent.c_sent.tokens[j].label)
            elif sent.c_sent.tokens[j].head > j:
                left_labels.add(sent.c_sent.tokens[j].label)
            else:
                right_labels.add(sent.c_sent.tokens[j].label)
    return list(sorted(left_labels)), list(sorted(right_labels)), list(sorted(dfl_labels))


def get_templates(feats_str):
    '''Interpret feats_str, returning a list of template tuples. Each template
    is a tuple of numeric indices, referring to positions in the context
    array. See _parse_features.pyx for examples. The templates are applied by
    thinc.features.Extractor, which picks out the appointed values and hashes
    the resulting array, to produce a single feature code.
    '''
    match_feats = []
    # This value comes out of compile_time_options.pxi
    IF TRANSITION_SYSTEM == 'arc_eager':
        templates = _parse_features.arc_eager
    ELSE:
        templates = _parse_features.arc_hybrid
    if 'disfl' in feats_str:
        templates += _parse_features.disfl
        templates += _parse_features.new_disfl
        templates += _parse_features.suffix_disfl
        templates += _parse_features.extra_labels
        templates += _parse_features.clusters
        templates += _parse_features.edges
        templates += _parse_features.prev_next
        match_feats = _parse_features.match_templates()
    elif 'clusters' in feats_str:
        templates += _parse_features.clusters
    if 'bitags' in feats_str:
        templates += _parse_features.pos_bigrams()
    return templates


cdef class Parser:
    cdef object cfg
    cdef Pool _pool
    cdef Extractor extractor
    cdef LinearModel guide
    cdef Tagger tagger
    cdef Transition* moves
    cdef atom_t* _context
    cdef size_t nr_moves

    def __init__(self, model_dir):
        assert os.path.exists(model_dir) and os.path.isdir(model_dir)
        self.cfg = Config.read(model_dir, 'config')
        self.extractor = Extractor(get_templates(self.cfg.features))
        self._pool = Pool()
        self._context = <atom_t*>self._pool.alloc(_parse_features.context_size(),
                                                  sizeof(atom_t))

        if os.path.exists(pjoin(model_dir, 'labels')):
            index.hashes.load_label_idx(pjoin(model_dir, 'labels'))
        self.nr_moves = get_nr_moves(self.cfg.left_labels, self.cfg.right_labels,
                                     self.cfg.dfl_labels, self.cfg.use_break)
        self.moves = <Transition*>self._pool.alloc(self.nr_moves, sizeof(Transition))
        fill_moves(self.cfg.left_labels, self.cfg.right_labels, self.cfg.dfl_labels,
                   self.cfg.use_break, self.moves)
        
        self.guide = LinearModel(self.nr_moves, self.extractor.n_templ)
        if os.path.exists(pjoin(model_dir, 'model')):
            self.guide.load(pjoin(model_dir, 'model'))
        if os.path.exists(pjoin(model_dir, 'pos')):
            index.hashes.load_pos_idx(pjoin(model_dir, 'pos'))
        self.tagger = Tagger(model_dir)

    cpdef int parse(self, Input py_sent) except -1:
        '''Parse a sentence, setting heads, labels and tags in-place.

        Args:
            py_sent (Input): The sentence to be parsed.
        '''
        cdef Sentence* sent = py_sent.c_sent
        cdef Token* gold_parse = sent.tokens
        if self.tagger:
            self.tagger.tag(py_sent)
        cdef Beam beam = Beam(self.nr_moves, self.cfg.beam_width)
        beam.initialize(_init_callback, sent.n, sent)
        self.guide.cache.flush()
        cdef int i
        while not beam.is_done:
            self._advance_beam(beam, NULL, False)
        _fill_parse(sent.tokens, <State*>beam.at(0))
        sent.score = beam.score

    cpdef int train_sent(self, Input py_sent) except -1:
        '''Receive a training example, and update weights if the prediction
        is incorrect.

        In order to account for "spurious ambiguity" in the transition system,
        we perform two searches given the current weights, constraining one so
        that the parser can only follow "zero-cost" transitions, where the cost
        is determined by an oracle which calculates the loss of the best parse
        reachable via that derivation. This allows us to find the best gold-standard
        derivation given our current weights.
        
        We then use the maximum violation strategy (Huang, 2012) to find a
        state-sequence to calculate the update.
        '''
        cdef Sentence* sent = py_sent.c_sent
        cdef Address tags_mem = Address(sent.n, sizeof(size_t))
        cdef size_t* gold_tags = <size_t*>tags_mem.ptr
        cdef Token* gold_parse = sent.tokens
        cdef int i
        for i in range(sent.n):
            gold_tags[i] = gold_parse[i].tag
        if self.tagger:
            self.tagger.tag(py_sent)
        cdef Beam p_beam = Beam(self.nr_moves, self.cfg.beam_width)
        cdef Beam g_beam = Beam(self.nr_moves, self.cfg.beam_width)
        p_beam.initialize(_init_callback, sent.n, sent)
        g_beam.initialize(_init_callback, sent.n, sent)

        cdef MaxViolation violn = MaxViolation()

        self.guide.cache.flush()
        cdef Transition* m
        cdef State* state
        while not p_beam.is_done and not g_beam.is_done:
            self._advance_beam(p_beam, gold_parse, False)
            self._advance_beam(g_beam, gold_parse, True)
            violn.check(p_beam, g_beam)
       
        is_true = p_beam._states[0].loss == 0
        counts = {}
        if not is_true:
            self._count_feats(counts, sent, violn.g_hist, 1)
            self._count_feats(counts, sent, violn.p_hist, -1)
            self.guide.update(counts)
        else:
            self.guide.update({})
        for i in range(sent.n):
            sent.tokens[i].tag = gold_tags[i]
        self.guide.n_corr += is_true
        self.guide.total += 1
        return is_true

    cdef int _advance_beam(self, Beam beam, Token* gold_parse, bint follow_gold) except -1:
        cdef int i, j
        for i in range(beam.size):
            state = <State*>beam.at(i)
            if is_final(state):
                continue
            if gold_parse != NULL:
                fill_costs(state, self.moves, self.nr_moves, gold_parse)
            if not follow_gold:
                fill_valid(state, self.moves, self.nr_moves)
            self._predict(state, self.moves)
            for j in range(self.nr_moves):
                m = &self.moves[j]
                beam.set_cell(i, j, m.score, m.is_valid, m.cost)
        beam.advance(_transition_callback, self.moves)
        beam.check_done(_is_done_callback, NULL)

    cdef int _predict(self, State* s, Transition* classes) except -1:
        if is_final(s):
            return 0
        fill_slots(s)
        fill_context(self._context, &s.slots, s.parse)
        cdef int n_feats = 0
        cdef Feature* feats = self.extractor.get_feats(self._context, &n_feats)
        cdef const weight_t* scores = self.guide.get_scores(feats, n_feats)
        cdef int i
        for i in range(self.nr_moves):
            classes[i].score = scores[i]

    cdef dict _count_feats(self, dict counts, Sentence* sent, list hist, int inc):
        cdef atom_t* context = self._context
        cdef Pool mem = Pool()
        cdef State* state = init_state(sent, mem)
        cdef class_t clas
        cdef int n_feats = 0
        cdef Feature* feats
        for clas in hist:
            fill_slots(state)
            fill_context(self._context, &state.slots, state.parse)
            feats = self.extractor.get_feats(context, &n_feats)
            count_feats(counts.setdefault(clas, {}), feats, n_feats, inc)
            transition(&self.moves[clas], state)


cdef int _fill_parse(Token* parse, State* s) except -1:
    cdef int i, head 
    for i in range(1, s.n-1):
        head = i
        while s.parse[head].head != head and \
                s.parse[head].head < (s.n-1) and \
                s.parse[head].head != 0:
            head = s.parse[head].head
        s.parse[i].sent_id = head
    # No need to copy heads for root and start symbols
    for i in range(1, s.n - 1):
        parse[i] = s.parse[i]


cdef void* _init_callback(Pool mem, int n, void* extra_args) except NULL:
    return init_state(<Sentence*>extra_args, mem)


cdef int _transition_callback(void* dest, void* src, class_t clas, void* extra_args) except -1:
    state = <State*>dest
    parent = <State*>src
    moves = <Transition*>extra_args
    copy_state(state, parent)
    transition(&moves[clas], state)


cdef int _is_done_callback(void* state, void* extra_args) except -1:
    return is_final(<State*>state)
