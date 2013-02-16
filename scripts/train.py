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

import redshift.parser
import redshift.io_parse

USE_HELD_OUT = False

@plac.annotations(
    train_loc=("Training location", "positional"),
    train_alg=("Learning algorithm [static, online]", "option", "a", str),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    label_set=("Name of label set to use.", "option", "l", str),
    add_extra_feats=("Add extra features", "flag", "x", bool),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    allow_reattach=("Allow left-clobber", "flag", "r", bool),
    allow_lower=("Allow raise/lower", "flag", "w", bool),
    allow_invert=("Allow invert", "flag", "v", bool),
    shiftless=("Use no shift transition (requires reattach)", "flag", "s", bool),
    repair_only=("Penalise incorrect moves in the oracle even when they can be repaired",
                 "flag", "o", bool),
    profile=("Run profiler (slow)", "flag", None, bool),
    debug=("Set debug flag to True.", "flag", None, bool),
    seed=("Set random seed", "option", None, int)
)
def main(train_loc, model_loc, train_alg="online", n_iter=15,
         add_extra_feats=False, label_set="Stanford", feat_thresh=1,
         allow_reattach=False, allow_lower=False, allow_invert=False,
         shiftless=False, repair_only=False,
         profile=False, debug=False, seed=0):
    random.seed(seed)
    train_loc = Path(train_loc)
    if shiftless:
        assert allow_reattach
    if repair_only:
        assert allow_reattach
    if allow_lower:
        assert allow_reattach
    model_loc = Path(model_loc)
    if label_set == 'None':
        label_set = None
    if debug:
        redshift.parser.set_debug(True)
    parser = redshift.parser.Parser(model_loc, clean=True,
                                    train_alg=train_alg, add_extra=add_extra_feats,
                                    label_set=label_set, feat_thresh=feat_thresh,
                                    allow_reattach=allow_reattach, allow_invert=allow_invert,
                                    allow_lower=allow_lower,
                                    shiftless=shiftless, repair_only=repair_only)
    if USE_HELD_OUT:
        train_sent_strs = train_loc.open().read().strip().split('\n\n')
        split_point = len(train_sent_strs)/20
        held_out = '\n\n'.join(train_sent_strs[:split_point])
        train = redshift.io_parse.read_conll('\n\n'.join(train_sent_strs[split_point:]))
        parser.train(train, held_out=held_out, n_iter=n_iter)
        to_parse = redshift.io_parse.read_conll('\n\n'.join(train_sent_strs[split_point:]))
    else:
        train = redshift.io_parse.read_conll(train_loc.open().read())
        if profile:
            print 'profiling'
            cProfile.runctx("parser.train(train, n_iter=n_iter)", globals(), locals(), "Profile.prof")
            s = pstats.Stats("Profile.prof")
            s.strip_dirs().sort_stats("time").print_stats()

        else:
            parser.train(train, n_iter=n_iter)
        to_parse = redshift.io_parse.read_conll(train_loc.open().read())
    print 'Train accuracy:'
    print parser.add_parses(to_parse, gold=train)
    parser.save()


if __name__ == "__main__":
    plac.call(main)
