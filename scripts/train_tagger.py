import plac

import redshift.tagger
import redshift.io_parse

@plac.annotations(
    iters=("Number of training iterations", "option", "i", int),
)
def main(model_loc, train_loc, test_loc, iters=10):
    tagger = redshift.tagger.BeamTagger(model_loc, clean=True)
    sents = redshift.io_parse.read_conll(open(train_loc).read())
    tagger.train(sents, nr_iter=iters)
    tagger.save()
    test_data = open(test_loc).read()
    to_tag = redshift.io_parse.read_conll(test_data)
    tagger.add_tags(to_tag)
    gold = redshift.io_parse.read_conll(test_data)
    acc, c, n = redshift.io_parse.eval_tags(to_tag, gold)
    print '%.2f' % acc, c, n

if __name__ == '__main__':
    plac.call(main)
