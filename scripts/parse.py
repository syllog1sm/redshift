#!/usr/bin/env python
#PBS -l walltime=1:00:00,mem=4gb,nodes=1:ppn=2
import os
import sys
import plac
from pathlib import Path
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
    parser_dir = Path(parser_dir)
    text_loc = Path(text_loc)
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()
    yield "Loading parser"
    parser = redshift.parser.Parser(parser_dir)
    parser.load()
    if use_gold:
        gold_sents = redshift.io_parse.read_conll(text_loc.open().read())
        pos_tagged = get_pos(text_loc.open().read())
        sentences = redshift.io_parse.read_pos(pos_tagged)
    else:
        sentences = redshift.io_parse.read_pos(text_loc.open().read())
        gold_sents = None
    if profile:
        cProfile.runctx("parser.add_parses(sentences,gold=gold_sents)", globals(), locals(), "Profile.prof")

        s = pstats.Stats("Profile.prof")
        s.strip_dirs().sort_stats("time").print_stats()
    else:

        t1 = time.time()
        print parser.add_parses(sentences, gold=gold_sents)
        t2 = time.time()
        print '%d sents took %0.3f ms' % (sentences.length, (t2-t1)*1000.0)
    
    sentences.write_moves(out_dir.join('moves').open('w'))
    sentences.write_parses(out_dir.join('parses').open('w'))
    if gold_sents:
        best_moves = parser.get_best_moves(sentences, gold_sents)
        with out_dir.join('best_moves').open('w') as out:
            for sent_str, moves in best_moves:
                out.write(sent_str + u'\n')
                for o_id, p_id, o_str, p_str, s_str in moves:
                    out.write(u'%s\t%s\t%s\n' % (o_str, p_str, s_str))
                out.write(u'\n')
                


if __name__ == '__main__':
    plac.call(main)
