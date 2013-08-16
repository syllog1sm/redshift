#!/usr/bin/env python

import plac
import time

import redshift.tagger
import redshift.io_parse

@plac.annotations(
    iters=("Number of training iterations", "option", "i", int),
    n_sents=("Number of training sentences", "option", "n", int),
    conll_format=("Read train and test in conll format", "flag", "c", bool)
)
def main(model_loc, train_loc, test_loc, iters=5, n_sents=0, conll_format=False):
    tagger = redshift.tagger.GreedyTagger(model_loc, clean=True)
    if conll_format:
        train_strs = open(train_loc).read().strip().split('\n\n')
    else:
        train_strs = open(train_loc).read().strip().split('\n')
    # Apply limit
    train_strs = train_strs[-n_sents:]
    if conll_format:
        train = redshift.io_parse.read_conll('\n\n'.join(train_strs))
    else:
        train = redshift.io_parse.read_pos('\n'.join(train_strs), sep='|')
    tagger.train(train, nr_iter=iters)
    tagger.save()
    test_data = open(test_loc).read()
    if conll_format:
        to_tag = redshift.io_parse.read_conll(test_data)
    else:
        to_tag = redshift.io_parse.read_pos(test_data, sep='|')
    t1 = time.time()
    tagger.add_tags(to_tag)
    t2 = time.time()
    print '%d sents took %0.3f ms' % (to_tag.length, (t2-t1)*1000.0)
    if conll_format:
        gold = redshift.io_parse.read_conll(test_data)
    else:
        gold = redshift.io_parse.read_pos(test_data, sep='|')
    acc, c, n = redshift.io_parse.eval_tags(to_tag, gold)
    print '%.2f' % acc, c, n

if __name__ == '__main__':
    plac.call(main)
