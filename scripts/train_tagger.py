#!/usr/bin/env python

import plac
import time

import redshift.tagger
import redshift.io_parse

@plac.annotations(
    iters=("Number of training iterations", "option", "i", int),
    n_sents=("Number of training sentences", "option", "n", int),
    feat_thresh=("Threshold for feature pruning", "option", "f", int),
)
def main(model_loc, train_loc, iters=5, n_sents=0, feat_thresh=5):
    tagger = redshift.tagger.GreedyTagger(model_loc, clean=True, feat_thresh=feat_thresh)
    train_strs = open(train_loc).read().strip().replace('|', '/').split('\n')
    # Apply limit
    if n_sents:
        train_strs = train_strs[:n_sents]
    tagger.train('\n'.join(train_strs), nr_iter=iters)
    tagger.save()

if __name__ == '__main__':
    plac.call(main)
