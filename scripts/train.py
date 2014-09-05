#!/usr/bin/env python
from __future__ import unicode_literals

import plac
import codecs

import redshift.parser
from redshift.sentence import Input


@plac.annotations(
    train_loc=("Training location", "positional"),
    codec=("Codec to be used to read training file", "option", "c", str),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    debug=("Set debug flag to True.", "flag", None, bool),
    beam_width=("Beam width", "option", "k", int),
    feat_set=("Name of feat set [zhang, iso, full]", "option", "x", str),
    n_sents=("Number of sentences to train from", "option", "n", int),
    use_break=("Use the Break transition", "flag", "b", bool),
    seed=("Random seed", "option", "s", int)
)
def main(train_loc, model_loc, n_iter=15,
         codec="utf8",
         feat_set="", feat_thresh=10,
         n_sents=0,
         use_break=False,
         debug=False, seed=0, beam_width=4,
         train_tagger=False):
    if debug:
        redshift.parser.set_debug(True)
    with codecs.open(train_loc, 'r', codec) as file_:
        train_str = file_.read()
    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        train_str = '\n\n'.join(train_str.split('\n\n')[:n_sents])
    redshift.parser.train(train_str.encode(codec), model_loc,
        n_iter=n_iter,
        beam_width=beam_width,
        feat_set=feat_set,
        feat_thresh=feat_thresh,
        use_break=use_break,
    )


if __name__ == "__main__":
    plac.call(main)
