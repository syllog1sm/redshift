#!/usr/bin/env python

import plac
from pathlib import Path
import math
import cProfile
import pstats
import sys


import redshift.nbest_parser
from redshift.sentence import Input
from redshift.util import read_nbest, get_nbest_loc
from redshift.util import get_oracle_alignment


def profile_training(sents, nbests, model_loc, n_iter, beam_width, feat_set):
    cProfile.runctx(
        """redshift.nbest_parser.train(sents, nbests, model_loc,
            n_iter=n_iter,
            beam_width=beam_width,
            feat_set=feat_set,
        )""", globals(), locals()
    )

    s = pstats.Stats("/tmp/Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
 


def get_nbest(gold_sent, nbest_dir, limit=0):
    # Need to copy the gold_sent, as we're going to pass gold_sent to the tagger
    # for training, and the tags get modified by nbest_train.
    gold_copy = Input.from_tokens([(t.word, t.tag, t.head, t.label, t.sent_id, t.is_edit)
                                    for t in gold_sent.tokens])
    nbest_loc = get_nbest_loc(gold_sent.turn_id, nbest_dir)
    if not nbest_loc:
        return [(1.0, gold_copy)]
    gold_tokens = list(gold_sent.tokens)
    gold_sent_id = gold_tokens[0].sent_id
    nbest = []
    seen_gold = False
    for prob, candidate in read_nbest(str(nbest_loc), limit=limit):
        cost, edits = get_oracle_alignment(candidate, gold_tokens)
        if cost == 0:
            sent = make_gold_sent(gold_tokens, candidate, edits)
            seen_gold = True
        else:
            sent = make_non_gold_sent(cost, candidate, gold_sent_id)
        nbest.append((prob, sent))
    # TODO: Should we always include this?
    if not seen_gold:
        nbest.append((0.0, gold_copy))
    return nbest


def make_non_gold_sent(wer, words, sent_id):
    tokens = [(word, None, None, None, sent_id, None) for word in words]
    return Input.from_tokens(tokens, wer=wer)


def make_gold_sent(gold, candidate, edits):
    tokens = []
    words = list(candidate)
    gold = list(gold)
    g_i = 0
    c_i = 0
    alignment = {}
    for op in edits:
        if op == 'M':
            alignment[g_i] = c_i
            g_i += 1
            c_i += 1
        elif op == 'I':
            g_i += 1
        elif op == 'D':
            c_i += 1
        else:
            raise StandardError(op)
    g_i = 0
    c_i = 0
    sent_id = gold[0].sent_id
    tokens = []
    for op in edits:
        if op == 'M':
            tokens.append(_make_token(candidate[c_i], gold[g_i], alignment))
            g_i += 1
            c_i += 1
        elif op == 'I':
            g_i += 1
        elif op == 'D':
            tokens.append(_make_dfl_token(candidate, c_i, sent_id))
            c_i += 1
    return Input.from_tokens(tokens)

def _make_token(word, gold, alignment):
    head = alignment.get(gold.head - 1, -1) + 1
    return (word, gold.tag, head, gold.label,
            gold.sent_id, gold.is_edit)

def _make_dfl_token(words, i, sent_id):
    word = words[i]
    last_word = words[i - 1] if i != 0 else 'EOL'
    next_word = words[i + 1] if i < (len(words) - 1) else 'EOL'
    return (word, 'UH', i + 1, _guess_label(word, last_word, next_word), sent_id, True)


def _guess_label(word, last_word, next_word):
    """
    13117 uh
    7189 you
    7186 know
    4569 well
    3633 oh
    3319 um
    1712 i
    1609 mean
    1050 like
    522 so
    372 huh
    284 now
    213 see
    124 yeah
    108 or
    106 actually
    """
    fillers = set(['uh', 'um', 'uhhuh', 'uh-huh', 'huh', 'oh'])
    discourse = set(['well', 'okay', 'actually', 'like', 'so', 'now', 'yeah'])
    editing = set(['or'])
    if word in fillers:
        return 'fillerF'
    elif word in discourse:
        return 'fillerD'
    elif word in editing:
        return 'fillerE'
    elif word == 'you' and next_word == 'know':
        return 'fillerD'
    elif word == 'know' and last_word == 'you':
        return 'fillerD'
    elif word == 'i' and next_word == 'mean':
        return 'fillerD'
    elif word == 'mean' and last_word == 'i':
        return 'fillerE'
    else:
        return 'erased'

@plac.annotations(
    train_loc=("Training location", "positional"),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    debug=("Set debug flag to True.", "flag", None, bool),
    beam_width=("Beam width", "option", "k", int),
    feat_set=("Name of feat set [zhang, iso, full]", "option", "x", str),
    n_sents=("Number of sentences to train from", "option", "n", int),
    limit=("Limit nbest list to N", "option", "N", int),
    use_break=("Use the Break transition", "flag", "b", bool),
    seed=("Random seed", "option", "s", int),
    profile=("Profile run-time", "flag", "p", bool)
)
def main(train_loc, nbest_dir, model_loc, n_iter=15,
         feat_set="disfl", feat_thresh=10,
         n_sents=0,
         limit=0,
         use_break=False,
         debug=False, seed=0, beam_width=4, profile=False):
    nbest_dir = Path(nbest_dir)
    if debug:
        redshift.parser.set_debug(True)
    train_str = open(train_loc).read()

    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        train_str = '\n\n'.join(train_str.split('\n\n')[:n_sents])
    sents = [Input.from_conll(s) for s in
             train_str.strip().split('\n\n') if s.strip()]
    nbests = [get_nbest(sent, nbest_dir, limit=limit) for sent in sents]
    if profile:
        profile_training(sents, nbests, model_loc, beam_width, n_iter, feat_set)
        sys.exit(1)

    print "Train"
    redshift.nbest_parser.train(sents, nbests, model_loc,
        n_iter=n_iter,
        beam_width=beam_width,
        feat_set=feat_set,
        feat_thresh=feat_thresh,
        use_break=use_break,
    )


if __name__ == "__main__":
    plac.call(main)
