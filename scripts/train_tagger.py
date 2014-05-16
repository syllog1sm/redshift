#!/usr/bin/env python

import plac
import time

import redshift.tagger
from redshift.sentence import Input

@plac.annotations(
    iters=("Number of training iterations", "option", "i", int),
    n_sents=("Number of training sentences", "option", "n", int),
    feat_thresh=("Threshold for feature pruning", "option", "f", int),
    beam_width=("Number of hypotheses to keep alive", "option", "k", int)
)
def main(model_dir, train_loc, dev_loc, iters=5, n_sents=0, feat_thresh=5, beam_width=4):
    sent_strs = open(train_loc).read().strip().replace('|', '/').split('\n')
    # Apply limit
    if n_sents != 0:
        sent_strs = sent_strs[:n_sents]
    tagger = redshift.tagger.train('\n'.join(sent_strs), model_dir,
        beam_width=beam_width, nr_iter=iters, feat_thresh=feat_thresh)
    dev_input = [Input.from_pos(s.replace('|', '/'))
                 for s in open(dev_loc).read().strip().split('\n')]
    t = 1e-100
    c = 0
    for sent in dev_input:
        gold_tags = [tok.tag for tok in sent.tokens]
        tagger.tag(sent)
        for i, token in enumerate(sent.tokens):
            c += gold_tags[i] == token.tag
            t += 1
    print c / t

if __name__ == '__main__':
    plac.call(main)
