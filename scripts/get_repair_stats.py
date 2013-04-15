#!/usr/bin/env python
"""
Parse a best_moves file to get repair stats.
"""
import plac
from collections import defaultdict

def sort_dict(d):
    return reversed(sorted(d.items(), key=lambda i: i[1]))

@plac.annotations(
    labels=("Print labelled moves", "flag", "l", bool)
)
def main(loc, labels=False):
    true_pos = defaultdict(int)
    false_pos = defaultdict(int)
    false_neg = defaultdict(int)
    true_neg = defaultdict(int)
    for line in open(loc):
        if '<start>' in line:
            continue
        line = line.rstrip()
        if not line:
            continue
        pieces = line.split('\t')
        golds = pieces[0].split(',')
        parse = pieces[1]
        is_punct = any(g.endswith('-P') for g in golds)
        if is_punct:
            continue
        if not labels:
            parse = parse.split('-')[0]
            golds = [g.split('-')[0] for g in golds]
        if len(golds) > 1:
            continue
        gold = golds[0]
        if gold.split('-')[0] == parse.split('-')[0]:
            if '^' in gold:
                true_pos[gold] += 1
            elif gold.startswith('L') or gold.startswith('D'):
                true_neg[gold] += 1
        else:
            if '^' in gold:
                false_neg[gold] += 1
            elif '^' in parse:
                false_pos[parse] += 1
    for label, d in [('TP', true_pos), ('FP', false_pos), ('FN', false_neg), ('TN', true_neg)]:
        print label
        for tag, freq in sort_dict(d):
            print tag, freq


if __name__ == '__main__':
    plac.call(main)
