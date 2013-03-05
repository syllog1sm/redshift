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
    train_alg=("Learning algorithm [static, online, beam]", "option", "a", str),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    label_set=("Name of label set to use.", "option", "l", str),
    add_extra_feats=("Add extra features", "flag", "x", bool),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    allow_reattach=("Allow left-clobber", "flag", "r", bool),
    allow_reduce=("Allow reduce when no head is set", "flag", "d", bool),
    profile=("Run profiler (slow)", "flag", None, bool),
    debug=("Set debug flag to True.", "flag", None, bool),
    seed=("Set random seed", "option", "s", int),
    beam_width=("Beam width", "option", "k", int),
    movebeam=("Add labels to beams", "flag", "m", bool),
    upd_strat=("Strategy for global updates [early, late, max]", "option", "u")
)
def main(train_loc, model_loc, train_alg="online", n_iter=15,
         add_extra_feats=False, label_set="Stanford", feat_thresh=1,
         allow_reattach=False, allow_reduce=False,
         profile=False, debug=False, seed=0, beam_width=1, movebeam=False,
         upd_strat="early"):
    random.seed(seed)
    train_loc = Path(train_loc)
    model_loc = Path(model_loc)
    if label_set == 'None':
        label_set = None
    if debug:
        redshift.parser.set_debug(True)
    parser = redshift.parser.Parser(model_loc, clean=True,
                                    train_alg=train_alg, add_extra=add_extra_feats,
                                    label_set=label_set, feat_thresh=feat_thresh,
                                    allow_reattach=allow_reattach, allow_reduce=allow_reduce,
                                    beam_width=beam_width, label_beam=not movebeam,
                                    upd_strat=upd_strat)
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
