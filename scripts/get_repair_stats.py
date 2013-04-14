"""
Parse a best_moves file to get repair stats.
"""
import plac
from collections import defaultdict

def sort_dict(d):
    return reversed(sorted(d.items(), key=lambda i: i[1]))

def main(loc):
    true_pos = defaultdict(int)
    false_pos = defaultdict(int)
    false_neg = defaultdict(int)
    for line in open(loc):
        if '<start>' in line:
            continue
        line = line.strip()
        if not line:
            continue
        pieces = line.split()
        golds = pieces[0].split(',')
        parse = pieces[1]
        if len(golds) == 1:
            gold = golds[0]
            if gold.endswith('-P'):
                continue
            if gold == parse:
                true_pos[gold] += 1
            else:
                false_neg[gold] += 1
        if parse not in golds and not parse.endswith('-P'):
            false_pos[parse] += 1
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
