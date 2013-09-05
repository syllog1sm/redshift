"""Do cross-fold/"jack knife" tagging of a training set."""

import plac
import time

import redshift.tagger
import redshift.io_parse

def train_and_tag(tagger, train_strs, test_strs):
    train_sents = redshift.io_parse.read_conll(train_strs)
    test_sents = redshift.io_parse.read_conll(test_strs)
    tagger.train(train_sents)
    tagger.add_tags(test_sents)
    return test_sents.write_pos()

@plac.annotations(
    iters=("Number of training iterations", "option", "i", int),
    n_splits=("Number of times to fold the data", "option", "n", int)
)
def main(model_loc, train_loc, iters=5, n_splits=10):
    tagger = redshift.tagger.BeamTagger(model_loc, clean=True)
    train_strs = open(train_loc).read().strip().split('\n')
    fold_size = len(train_strs) / n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = test_start + fold_size
        test_strs = '\n'.join(train_strs[test_start:test_end])
        train_strs = '\n'.join(train_strs[:test_start] + train_strs[test_end:])
        print train_and_tag(tagger, train_strs, test_strs)    

if __name__ == '__main__':
    plac.call(main)
