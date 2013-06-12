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
    feat_thresh=("Feature pruning threshold", "option", "t", int),
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
         add_extra_feats=False, label_set="Stanford", feat_thresh=1,
         allow_reattach=False, allow_reduce=False, ngrams='best',
         add_clusters=False, n_sents=0,
         profile=False, debug=False, seed=0, beam_width=1, movebeam=False):
    best_bigrams = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,32,33,34,36,38,40,41,43,45,47,50,51,52,54,55,56,57,58,59,60,61,62,63,64]
    n_kernel_tokens = len(redshift.features.get_kernel_tokens())
    n_bigrams = len(list(combinations(range(n_kernel_tokens), 2)))
    n_ngrams = n_bigrams + len(list(combinations(range(n_kernel_tokens), 3)))
    if ngrams == 'all_bi':
        ngrams = range(n_bigrams)
    elif ngrams == 'base':
        ngrams = []
    elif ngrams == 'best':
        ngrams = best_bigrams
    elif ngrams.startswith('in'):
        ngrams = [int(ngrams[2:])]
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
    parser = redshift.parser.Parser(model_loc, clean=True,
                                    train_alg=train_alg, add_extra=add_extra_feats,
                                    label_set=label_set, feat_thresh=feat_thresh,
                                    allow_reattach=allow_reattach, allow_reduce=allow_reduce,
                                    beam_width=beam_width, label_beam=not movebeam,
                                    ngrams=ngrams, add_clusters=add_clusters)
    
    train_sent_strs = train_loc.open().read().strip().split('\n\n')
    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        random.shuffle(train_sent_strs)
        train_sent_strs = train_sent_strs[:n_sents]
    train_str = '\n\n'.join(train_sent_strs)
    train = redshift.io_parse.read_conll(train_str)
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
