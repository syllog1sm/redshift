import plac
import time

import redshift.tagger
import redshift.io_parse

def main(model_loc, test_loc):
    tagger = redshift.tagger.GreedyTagger(model_loc, trained=True)
    test_data = open(test_loc).read()
    for sent_str in open(test_loc):
        tagged = tagger.tag(sent_str, tokenize=False)
        print ' '.join('|'.join(tok) for tok in tagged)

if __name__ == '__main__':
    plac.call(main)
