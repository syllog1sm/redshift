#!/usr/bin/env python
#PBS -l walltime=10:00:00,mem=10gb,nodes=1:ppn=6

import os
import sys
import plac
import time
from pathlib import Path


import redshift.parser
import redshift.io_parse

@plac.annotations(
    train_loc=("Training location", "positional"),
    moves_loc=("Training moves location", "positional"),
    train_alg=("Learning algorithm [static, online]", "option", "a", str),
    label_set=("Name of label set to use.", "option", "l", str),
    add_extra_feats=("Add extra features", "flag", "x", bool),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    allow_reattach=("Allow left-clobber", "flag", "r", bool),
    allow_lower=("Allow raise/lower", "flag", "w", bool),
    shiftless=("Use no shift transition (requires reattach)", "flag", "s", bool),
    repair_only=("Penalise incorrect moves in the oracle even when they can be repaired",
                 "flag", "o", bool),
)
def main(train_loc, model_loc, moves_loc=None, train_alg="static",
         add_extra_feats=False, label_set="Stanford", feat_thresh=1,
         allow_reattach=False, allow_lower=False, shiftless=False,
         repair_only=False):
    train_loc = Path(train_loc)
    if allow_reattach:
        grammar_loc = train_loc.parent().join('rgrammar')
    else:
    	grammar_loc = None
    if shiftless:
        assert allow_reattach
    model_loc = Path(model_loc)
    if label_set == 'None':
        label_set = None
    if moves_loc is not None:
        moves_loc = Path(moves_loc)
        if not moves_loc.exists():
            print "Could not find moves; assuming none"
            moves_loc = None
    parser = redshift.parser.Parser(model_loc, clean=True,
                                    train_alg=train_alg, add_extra=add_extra_feats,
                                    label_set=label_set, feat_thresh=feat_thresh,
                                    allow_reattach=allow_reattach, allow_lower=allow_lower,
                                    grammar_loc=grammar_loc, shiftless=shiftless,
                                    repair_only=repair_only)
    if moves_loc is not None:
        moves = moves_loc.open().read().strip()
    else:
        moves = None
    train = redshift.io_parse.read_conll(train_loc.open().read(), moves=moves)
    parser.train(train)
    print 'Train accuracy:'
    to_parse = redshift.io_parse.read_conll(train_loc.open().read(), moves=moves)
    print parser.add_parses(to_parse, gold=train)
    parser.save()


if __name__ == "__main__":
    plac.call(main)
