#!/usr/bin/env python
import os
import os.path
import sys
import plac
import time
import pstats
import cProfile

import redshift.parser


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
    profile=("Do profiling", "flag", "p", bool),
    debug=("Set debug", "flag", "d", bool)
)
def main(parser_dir, text_loc, out_dir, profile=False, debug=False):
    if debug:
        redshift.parser.set_debug(debug)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    yield "Loading parser"
    parser = redshift.parser.load_parser(parser_dir)
    #sentences.connect_sentences(1700)
    if profile:
        cProfile.runctx("parser.add_parses(sentences)",
                        globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        t1 = time.time()
        parser.parse_file(text_loc, os.path.join(out_dir, 'parses'))
        t2 = time.time()
        print 'Parsing took %0.3f ms' % ((t2-t1)*1000.0)


if __name__ == '__main__':
    plac.call(main)
