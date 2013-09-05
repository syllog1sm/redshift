"""
Gather statistics about parser behaviour for disfluency detection
"""
from redshift.parser import load_parser
from redshift.io_parse import read_conll
from redshift.parser import get_edit_stats

import plac

def main(model_loc, dev_loc):
    parser = load_parser(model_loc)
    sents = read_conll(open(dev_loc).read())
    get_edit_stats(parser, sents)


if __name__ == '__main__':
    plac.call(main)
