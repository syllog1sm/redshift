#!/usr/bin/env python

import plac

import redshift.parser


@plac.annotations(
    train_loc=("Training location", "positional"),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    debug=("Set debug flag to True.", "flag", None, bool),
    beam_width=("Beam width", "option", "k", int),
    feat_set=("Name of feat set [zhang, iso, full]", "option", "x", str),
    n_sents=("Number of sentences to train from", "option", "n", int),
    train_tagger=("Train tagger alongside parser", "flag", "p", bool)
)
def main(train_loc, model_loc, n_iter=15,
         feat_set="zhang", feat_thresh=10,
         n_sents=0,
         debug=False, seed=0, beam_width=1,
         train_tagger=False):
    if debug:
        redshift.parser.set_debug(True)
    #if beam_width >= 2:
    train_str = open(train_loc).read()
    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        train_str = '\n\n'.join(train_str.split('\n\n')[:n_sents])
    redshift.parser.train(train_str, model_loc,
        n_iter=n_iter,
        train_tagger=train_tagger,
        beam_width=beam_width,
        feat_set=feat_set,
        feat_thresh=feat_thresh
    )


if __name__ == "__main__":
    plac.call(main)
