#!/usr/bin/env python
import os
import os.path
import sys
import plac
import time
import pstats
import cProfile

import redshift.parser
from redshift.sentence import Input


def parse(parser, sentences):
    for sent in sentences:
        parser.parse(sent)


@plac.annotations(
    profile=("Do profiling", "flag", "p", bool),
    debug=("Set debug", "flag", "d", bool)
)
def main(parser_dir, text_loc, out_dir, profile=False, debug=False):
    if debug:
        redshift.parser.set_debug(debug)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print "Loading parser"
    parser = redshift.parser.Parser(parser_dir)
    sentences = [Input.from_pos(p.strip()) for i, p in
                 enumerate(open(text_loc).read().strip().split('\n'))]
    if profile:
        cProfile.runctx("parse(parser, sentences)",
                        globals(), locals(), "Profile.prof")
        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:
        t1 = time.time()
        parse(parser, sentences)
        t2 = time.time()
        print '%d sents took %0.3f ms' % (len(sentences), (t2-t1)*1000.0)

    with open(os.path.join(out_dir, 'parses'), 'w') as out_file:
        for sentence in sentences:
            out_file.write(sentence.to_conll())
            out_file.write('\n\n')


if __name__ == '__main__':
    plac.call(main)
