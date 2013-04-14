#!/usr/bin/env python
"""
Parse a best_moves file to get repair stats.
"""
import plac
from collections import defaultdict

def sort_dict(d):
    return reversed(sorted(d.items(), key=lambda i: i[1]))

@plac.annotations(
    repairs=("Only print for repair moves", "flag", "r", bool),
    labels=("Print labelled moves", "flag", "l", bool)
)
def main(loc, repairs=False, labels=False):
    true_pos = defaultdict(int)
    false_pos = defaultdict(int)
    false_neg = defaultdict(int)
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
        if parse not in golds and len(golds) == 1:
            if (not repairs or '^' in parse):
                false_pos[parse] += 1
        if len(golds) == 1:
            gold = golds[0]
            if gold == parse:
                if (not repairs or '^' in gold):
                    true_pos[gold] += 1
            else:
                if (not repairs or '^' in gold):
                    false_neg[gold] += 1
    print 'TP'
    for tag, freq in sort_dict(true_pos):
        print freq, tag
    print sum(true_pos.values())
    print 'FP'
    for tag, freq in sort_dict(false_pos):
        print freq, tag
    print sum(false_pos.values())
    print 'FN'
    for tag, freq in sort_dict(false_neg):
        print freq, tag
    print sum(false_neg.values())


if __name__ == '__main__':
    plac.call(main)
