#!/usr/bin/env python
import os
import os.path
import sys
import plac
import time
import pstats
import cProfile

import redshift.parser
import redshift.io_parse


def get_pos(conll_str):
    pos_sents = []
    for sent_str in conll_str.strip().split('\n\n'):
        sent = []
        for line in sent_str.split('\n'):
            pieces = line.split()
            if len(pieces) == 5:
                pieces.pop(0)
            word = pieces[0]
            pos = pieces[1]
            sent.append('%s/%s' % (word, pos))
        pos_sents.append(' '.join(sent))
    return '\n'.join(pos_sents)


@plac.annotations(
    use_gold=("Gold-formatted test data", "flag", "g", bool),
    profile=("Do profiling", "flag", "p", bool),
    debug=("Set debug", "flag", "d", bool)
)
def main(parser_dir, text_loc, out_dir, use_gold=False, profile=False, debug=False):
    if debug:
        redshift.parser.set_debug(debug)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    yield "Loading parser"
    parser = redshift.parser.load_parser(parser_dir)
    sentences = redshift.io_parse.read_pos(open(text_loc).read())
    #sentences.connect_sentences(1700)
    if profile:
        cProfile.runctx("parser.add_parses(sentences,gold=gold_sents)",
                        globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        t1 = time.time()
        parser.add_parses(sentences)
        t2 = time.time()
        print '%d sents took %0.3f ms' % (sentences.length, (t2-t1)*1000.0)
    sentences.write_parses(open(os.path.join(out_dir, 'parses'), 'w'))


if __name__ == '__main__':
    plac.call(main)
