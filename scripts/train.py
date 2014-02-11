#!/usr/bin/env python

import random
import os
import sys
import plac
import time
try:
    import pstats
    import cProfile
except ImportError:
    pass
from itertools import combinations

import redshift.parser
from redshift.parser import GreedyParser, BeamParser
from redshift import Sentence


@plac.annotations(
    train_loc=("Training location", "positional"),
    train_alg=("Learning algorithm [static, dyn]", "option", "a", str),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    vocab_thresh=("Vocab pruning threshold", "option", "t", int),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    allow_reattach=("Allow left-clobber", "flag", "r", bool),
    allow_reduce=("Allow reduce when no head is set", "flag", "d", bool),
    use_edit=("Use edit transition", "flag", "e", bool),
    profile=("Run profiler (slow)", "flag", None, bool),
    debug=("Set debug flag to True.", "flag", None, bool),
    seed=("Set random seed", "option", "s", int),
    beam_width=("Beam width", "option", "k", int),
    feat_set=("Name of feat set [zhang, iso, full]", "option", "x", str),
    n_sents=("Number of sentences to train from", "option", "n", int),
    auto_pos=("Train tagger alongside parser", "flag", "p", bool)
)
def main(train_loc, model_loc, train_alg="static", n_iter=15,
         feat_set="zhang", vocab_thresh=0, feat_thresh=10,
         allow_reattach=False, allow_reduce=False, use_edit=False,
         n_sents=0,
         profile=False, debug=False, seed=0, beam_width=1, unlabelled=False,
         auto_pos=False):
    random.seed(seed)
    if debug:
        redshift.parser.set_debug(True)
    if beam_width >= 2:
        parser = BeamParser(model_loc, clean=True, use_edit=use_edit,
                            train_alg=train_alg, feat_set=feat_set,
                            feat_thresh=feat_thresh, allow_reduce=allow_reduce,
                            allow_reattach=allow_reattach, beam_width=beam_width,
                            auto_pos=auto_pos)
    else:
        parser = GreedyParser(model_loc, clean=True, train_alg=train_alg,
                              feat_set=feat_set, feat_thresh=feat_thresh,
                              allow_reduce=allow_reduce,
                              allow_reattach=allow_reattach, use_edit=use_edit,
                              auto_pos=auto_pos)
    train_sent_strs = open(train_loc).read().strip().split('\n\n')
    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        random.shuffle(train_sent_strs)
        train_sent_strs = train_sent_strs[:n_sents]
    train = [Sentence.from_conll(i, s) for i, s in enumerate(train_sent_strs)]
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
