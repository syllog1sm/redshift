#!/usr/bin/env python
#PBS -l walltime=10:00:00,mem=10gb,nodes=1:ppn=6

import random
import os
import sys
import plac
import time
from pathlib import Path
import pstats
import cProfile
from itertools import combinations

import redshift.parser
import redshift.io_parse
import redshift.features

USE_HELD_OUT = False

@plac.annotations(
    train_loc=("Training location", "positional"),
    train_alg=("Learning algorithm [static, online, max, early]", "option", "a", str),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    label_set=("Name of label set to use.", "option", "l", str),
    add_extra_feats=("Add extra features", "flag", "x", bool),
    vocab_thresh=("Vocab pruning threshold", "option", "t", int),
    allow_reattach=("Allow left-clobber", "flag", "r", bool),
    allow_reduce=("Allow reduce when no head is set", "flag", "d", bool),
    profile=("Run profiler (slow)", "flag", None, bool),
    debug=("Set debug flag to True.", "flag", None, bool),
    seed=("Set random seed", "option", "s", int),
    beam_width=("Beam width", "option", "k", int),
    movebeam=("Add labels to beams", "flag", "m", bool),
    ngrams=("What ngrams to include", "option", "b", str),
    add_clusters=("Add brown cluster features", "flag", "c", bool),
    n_sents=("Number of sentences to train from", "option", "n", int)
)
def main(train_loc, model_loc, train_alg="online", n_iter=15,
         add_extra_feats=False, label_set="Stanford", vocab_thresh=0,
         allow_reattach=False, allow_reduce=False, ngrams='base',
         add_clusters=False, n_sents=0,
         profile=False, debug=False, seed=0, beam_width=1, movebeam=False):
    best_bigrams = [0, 26, 12, 126, 1, 5, 41, 16, 40, 86, 20, 87, 18, 27, 22, 30,
                    3, 104, 24, 65, 117, 132, 29, 11, 34, 131, 7, 116, 32, 36, 81,
                    15, 9, 21, 44, 6, 128, 95, 89, 17, 96, 38, 19, 84, 14, 43, 4,
                    2, 82, 90, 54, 76, 58, 77, 53, 23, 13, 31, 28, 42, 101, 35, 111,
                    121, 122, 25, 10, 127, 106, 129, 130, 33, 120, 37, 100, 66, 135,
                    59, 110, 8, 61, 107]
    best_trigrams = [69, 67, 71, 68, 66, 72, 73, 74, 70, 123, 138, 93, 78]
    n_kernel_tokens = len(redshift.features.get_kernel_tokens())
    bigrams = list(combinations(range(n_kernel_tokens), 2))
    n_bigrams = len(bigrams)
    trigrams = list(combinations(range(n_kernel_tokens), 3))
    if ngrams == 'base':
        ngrams = []
    elif ngrams == 'best':
        ngrams = best_bigrams
    elif ngrams.startswith('bi'):
        idx = int(ngrams[2:])
        ngrams = [idx]
    elif ngrams.startswith('tri'):
        idx = int(ngrams[3:])
        trigram = trigrams[idx]
        ngrams = [n_bigrams + idx,
                  bigrams.index((trigram[0], trigram[1])),
                  bigrams.index((trigram[0], trigram[2])),
                  bigrams.index((trigram[1], trigram[2]))]
    elif ngrams.startswith('btri'):
        idx = int(ngrams[4:])
        trigram = trigrams[idx]
        ngrams = [bigrams.index((trigram[0], trigram[1])),
                  bigrams.index((trigram[0], trigram[2])),
                  bigrams.index((trigram[1], trigram[2]))]

    else:
        raise StandardError, ngrams
    random.seed(seed)
    train_loc = Path(train_loc)
    model_loc = Path(model_loc)
    if label_set == 'None':
        label_set = None
    elif label_set == 'conll':
        label_set = str(train_loc)
    if debug:
        redshift.parser.set_debug(True)
    # TODO: In principle we don't need to ensure the vocab thresh is respected
    # at parse time, but we'll see about that...
    parser = redshift.parser.Parser(model_loc, clean=True,
                                    train_alg=train_alg, add_extra=add_extra_feats,
                                    label_set=label_set,
                                    allow_reattach=allow_reattach, allow_reduce=allow_reduce,
                                    beam_width=beam_width, label_beam=not movebeam,
                                    ngrams=ngrams, add_clusters=add_clusters)
    
    train_sent_strs = train_loc.open().read().strip().split('\n\n')
    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        random.shuffle(train_sent_strs)
        train_sent_strs = train_sent_strs[:n_sents]
    train_str = '\n\n'.join(train_sent_strs)
    train = redshift.io_parse.read_conll(train_str, vocab_thresh=vocab_thresh)
    #train.connect_sentences(1000)
    if profile:
        print 'profiling'
        cProfile.runctx("parser.train(train, n_iter=n_iter)", globals(),
                        locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        parser.train(train, n_iter=n_iter)
    parser.save()


if __name__ == "__main__":
    plac.call(main)
