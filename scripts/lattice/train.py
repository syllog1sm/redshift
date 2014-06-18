#!/usr/bin/env python

import plac
from pathlib import Path
import math
import cProfile
import pstats

import redshift.parser
from redshift.sentence import Input
from redshift.lattice_utils import read_lattice, add_gold_parse


def profile_training(sents, model_loc, n_iter, beam_width, beam_factor, feat_set, feat_thresh):
    cProfile.runctx("""redshift.parser.train(sents, model_loc,
                        n_iter=n_iter,
                        beam_width=beam_width,
                        lattice_factor=lattice_factor,
                        feat_set=feat_set,
                        feat_thresh=feat_thresh)""", globals(), locals(), 
                        "/tmp/Profile.prof"
                    )
    s = pstats.Stats("/tmp/Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()
 

def get_gold_lattice(conll_str, asr_dir, beta=0.0):
    gold_sent = Input.from_conll(conll_str)
    turn_id = gold_sent.turn_id
    filename, turn_num = gold_sent.turn_id.split('~')
    speaker = turn_num[0]
    turn_num = turn_num[1:]
    turn_id = '%s%s~%s' % (filename, speaker, turn_num)
    lattice_loc = asr_dir.joinpath(filename).joinpath(speaker).joinpath('raw').joinpath(turn_id)
    if lattice_loc.exists(): 
        lattice = read_lattice(str(lattice_loc), add_gold=True, beta=beta)
        add_gold_parse(lattice, gold_sent)
        return lattice
    else:
        return None
    

@plac.annotations(
    train_loc=("Training location", "positional"),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    beam_width=("Beam width", "option", "k", int),
    lattice_factor=("Prune lattice when lowest score * beta < max score.", "option", "b", float),
    feat_set=("Name of feat set [zhang, iso, full]", "option", "x", str),
    n_sents=("Number of sentences to train from", "option", "n", int),
    seed=("Random seed", "option", "s", int),
    profile=("Do profiling", "flag", "p", bool),
)
def main(train_loc, asr_dir, model_loc, n_iter=15,
         feat_set="disfl", feat_thresh=10,
         n_sents=0, seed=0, beam_width=4, lattice_factor=0.0,
         profile=False):
    asr_dir = Path(asr_dir)
    train_str = open(train_loc).read()

    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        train_str = '\n\n'.join(train_str.split('\n\n')[:n_sents])
    sents = [get_gold_lattice(s, asr_dir, beta=lattice_factor) for s in
             train_str.strip().split('\n\n') if s.strip()]
    sents = [s for s in sents if s is not None]
    if profile:
        profile_training(sents, model_loc, n_iter, beam_width, beam_factor, feat_set, feat_thresh)
    else:
        redshift.parser.train(sents, model_loc,
            n_iter=n_iter,
            beam_width=beam_width,
            lattice_factor=lattice_factor,
            feat_set=feat_set,
            feat_thresh=feat_thresh,
        )


if __name__ == "__main__":
    plac.call(main)
