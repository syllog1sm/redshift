#!/usr/bin/env python
from __future__ import unicode_literals
import os
import os.path
import sys
import plac
import time
import pstats
import cProfile
import codecs

import redshift.parser
from redshift.sentence import Input
try:
    import humanize
except ImportError:
    pass


def parse(parser, sentences):
    for sent in sentences:
        parser.parse(sent)


 
def mem(size="rss"):
    """Generalization; memory sizes: rss, rsz, vsz."""
    size = int(os.popen('ps -p %d -o %s | tail -1' %
                        (os.getpid(), size)).read())
    try:
        return humanize.naturalsize(size * 1024, gnu=True)
    except:
        return str(size)
 

@plac.annotations(
    profile=("Do profiling", "flag", "p", bool),
    codec=("Input codec", "option", "c", str),
    debug=("Set debug", "flag", "d", bool)
)
def main(parser_dir, text_loc, out_dir, codec="utf8", profile=False, debug=False):
    if debug:
        redshift.parser.set_debug(debug)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    print "Loading parser"
    parser = redshift.parser.Parser(parser_dir)
    for i in range(1000):
        with codecs.open(text_loc, 'r', 'utf8') as file_:
            input_text = file_.read()
            sentences = [Input.from_pos(p.strip().encode(codec)) for i, p in
                     enumerate(input_text.split('\n'))
                     if p.strip()]
            t1 = time.time()
            parse(parser, sentences)
            t2 = time.time()
            print '%d sents took %0.3f ms. %s mem' % (len(sentences), (t2-t1)*1000.0,
                                                      mem())

if __name__ == '__main__':
    plac.call(main)
