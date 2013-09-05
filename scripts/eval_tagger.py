import plac
import time

import redshift.tagger
import redshift.io_parse

def main(model_loc, test_loc):
    tagger = redshift.tagger.GreedyTagger(model_loc, trained=True)
    test_data = open(test_loc).read()
    to_tag = redshift.io_parse.read_pos(test_data, sep='|')
    t1 = time.time()
    tagger.add_tags(to_tag)
    t2 = time.time()
    print '%d sents took %0.3f ms' % (to_tag.length, (t2-t1)*1000.0)
    gold = redshift.io_parse.read_pos(test_data, sep='|')
    acc, c, n = redshift.io_parse.eval_tags(to_tag, gold)
    print '%.2f' % acc, c, n

if __name__ == '__main__':
    plac.call(main)
