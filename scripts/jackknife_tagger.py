"""Train a tagger on a folds directory, to produce jack-knifed input."""

from redshift.tagger import GreedyTagger
from redshift.io_parse import read_conll
import index.hashes

import plac
from pathlib import Path


def do_fold(train_loc, model_loc, test_loc, out_loc, iters=5):
    print model_loc, out_loc
    tagger = GreedyTagger(str(model_loc), clean=True, reuse_idx=True)
    train_strs = train_loc.open().read().strip().split('\n\n')

    train = read_conll('\n\n'.join(train_strs))
    tagger.train(train, nr_iter=iters)
    tagger.save()
    test_data = test_loc.open().read()
    to_tag = read_conll(test_data)
    tagger.add_tags(to_tag)
    to_tag.write_tags(open(str(out_loc), 'w'))

def main(folds_dir):
    folds_dir = Path(folds_dir)
    index.hashes.init_word_idx(str(folds_dir.join('words')))
    index.hashes.init_pos_idx(str(folds_dir.join('pos')))
    index.hashes.init_label_idx(str(folds_dir.join('labels')))
    for i in range(10):
        train_loc = folds_dir.join(str(i) + '.train')
        model_dir = folds_dir.join('model_%d' % i)
        test_loc = folds_dir.join(str(i) + '.test')
        out_loc = folds_dir.join(str(i) + '.tagged')
        do_fold(train_loc, model_dir, test_loc, out_loc)

if __name__ == '__main__':
    plac.call(main)

