#!/usr/bin/env python
import os
import os.path
import sys
import plac
import time
import pstats
import cProfile
from pathlib import Path
import math
import sh

import redshift.nbest_parser
from redshift.sentence import Input
from redshift.util import read_nbest, get_nbest_loc
from redshift.util import levenshtein
from redshift.util import get_oracle_alignment


def run_qian_dfl(strings):
    open('/tmp/for_dfl', 'w').write('\n'.join(strings))
    try:
        sh.qian('/tmp/for_dfl', '/tmp/dfl_out')
    except sh.CommandNotFound:
        return strings
    output = []
    for line in open('/tmp/dfl_out').read().strip().split('\n'):
        tokens = [t.rsplit('/', 3) for t in line.split()]
        fluent = [w for (w, p, d) in tokens if d == 'S-O' or (d == 'S-I' and p == 'CC')]
        output.append(fluent)
    return output


def get_nbest(turn_id, nbest_dir, limit=0):
    filename, turn_num = turn_id.split('~')
    speaker = turn_num[0]
    turn_num = turn_num[1:]
    turn_id = '%s%s~%s' % (filename, speaker, turn_num)
    nbest_loc = nbest_dir.join(filename).join(speaker).join('nbest').join(turn_id)
    if not nbest_loc.exists():
        return []
    else:
        return list(read_nbest(str(nbest_loc), limit))


def get_lower_bound(nbest, gold):
    candidates = [c for _, c in nbest]
    errors = [get_oracle_alignment(c, gold) for c in candidates]
    return min(errors)[0]


def get_verbatim_wer(candidate, gold):
    return levenshtein(candidate, [w.word for w in gold])


def sparseval(test, gold):
    test_deps = _get_deps(test)
    gold_deps = _get_deps(gold)
    tp = test_deps.intersection(gold_deps)
    fn = gold_deps - test_deps
    return len(tp), len(fn), len(test_deps)


def _get_deps(tokens):
    deps = set()
    n = len(tokens)
    for t in tokens:
        if t.is_edit:
            continue
        head = tokens[t.head - 1].word if (t.head - 1) < n else "ROOT"
        deps.add((t.word, head))
    return deps


@plac.annotations(
    mix_weight=("Control the LM and acoustic model mixture", "option", "w", float),
    limit=("Limit N-best to N candidates", "option", "N", int)
)
def main(parser_dir, conll_loc, nbest_dir, out_dir, mix_weight=0.0, limit=0):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    nbest_dir = Path(nbest_dir)
    print "Loading parser"
    parser = redshift.nbest_parser.NBestParser(parser_dir)
    gold_sents = [Input.from_conll(s) for s in
                  open(conll_loc).read().strip().split('\n\n') if s.strip()]
    wer = 0
    verbatim = 0
    verbatim_oracle = 0
    v_n = 0
    baseline = 0
    oracle = 0
    asr_fluent = 0
    n = 0
    tp_deps = 0
    fn_deps = 0
    n_deps = 0
    asr_strings = []
    fluent_strings = []
    for gold in gold_sents:
        nbest = read_nbest(get_nbest_loc(gold.turn_id, nbest_dir), limit)
        if not nbest:
            continue
        gold_tokens = list(gold.tokens)
        
        parses = parser.parse_nbest(nbest)
        first = parses[0]
        guess = max(parses, key=lambda sent: sent.score)
        
        fluent_gold = [t.word for t in gold.tokens if not t.is_edit]
        oracle += get_lower_bound(nbest, gold_tokens)
        asr_strings.append(' '.join(nbest[0][1]))
        fluent_strings.append(fluent_gold)
        verb_scores = [get_verbatim_wer(s, gold_tokens) for p, s in nbest]
        verbatim += verb_scores[0]
        verbatim_oracle += min(verb_scores)
        v_n += len(gold_tokens)


        parse_scores = sparseval(list(guess.tokens), gold_tokens)
        tp_deps += parse_scores[0]
        fn_deps += parse_scores[1]
        n_deps += parse_scores[2]
        fluent_guess = [t.word for t in guess.tokens if not t.is_edit]
        fluent_bl = [t.word for t in first.tokens if not t.is_edit]
        baseline += levenshtein(fluent_gold, fluent_bl)
        wer += levenshtein(fluent_guess, fluent_gold)
        asr_fluent += levenshtein(nbest[0][1], fluent_gold)
        n += len(fluent_gold)
    dfl_strings = run_qian_dfl(asr_strings)
    assert len(dfl_strings) == len(fluent_strings)
    qian_pl = sum(levenshtein(dfl, gold) for (dfl, gold) in zip(dfl_strings, fluent_strings))
    print "Against verbatim:"
    print '1-best: %.3f' % (float(verbatim) / v_n)
    print 'Oracle: %.3f' % (float(verbatim_oracle) / v_n)
    print "Against fluent:"
    print 'ASR: %.3f' % (float(asr_fluent) / n)
    print 'ASR --> Qian: %.3f' % (float(qian_pl) / n)
    print 'ASR --> dfl.: %.3f' % (float(baseline) / n)
    print 'Hyp: %.3f' % (float(wer) / n)
    print 'Oracle: %.3f' % (float(oracle) / n)
    print "Sparseval:"
    p = float(tp_deps) / n_deps
    r = float(tp_deps) / (tp_deps + fn_deps)
    print "P: %.3f" % p
    print "R: %.3f" % r
    print "F: %.3f" % ((2 * p * r) / (p + r))


if __name__ == '__main__':
    plac.call(main)
