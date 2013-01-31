#!/usr/bin/env python
import os
import sys
from pathlib import Path
import plac
from collections import defaultdict

def pc(num, den):
    return '%.2f' % ((num / float(den)) * 100)

@plac.annotations(
    eval_punct=("Evaluate punct transitions", "flag", "p")
)
def main(test_loc, gold_loc, eval_punct=False):
    test_loc = Path(test_loc)
    gold_loc = Path(gold_loc)
    if not test_loc.exists():
        test_loc.mkdir()
    test_sents = test_loc.open().read().strip().split('\n\n')
    gold_sents = gold_loc.open().read().strip().split('\n\n')
    assert len(test_sents) == len(gold_sents)
    n_by_label = defaultdict(int)
    u_by_label = defaultdict(int)
    l_by_label = defaultdict(int)
    for test_sent, gold_sent in zip(test_sents, gold_sents):
        test_sent = test_sent.split('\n')
        gold_sent = gold_sent.split('\n')
        assert len(test_sent) == len(gold_sent)
        for t, g in zip(test_sent, gold_sent):
            t = t.strip().split()
            g = g.strip().split()
            g_label = g[-1]
            if g_label == "P" and not eval_punct:
                continue
            t_label = t[-1]
            g_head = g[-2]
            t_head = t[-2]
            u_c = g_head == t_head
            l_c = u_c and g_label == t_label
            n_by_label[g_label] += 1
            u_by_label[g_label] += u_c
            l_by_label[g_label] += l_c
    for label, n in n_by_label.items():
        yield '%s\t%d\t%s\t%s' % (label, n, pc(l_by_label[label], n), pc(u_by_label[label], n))
    n = float(sum(n_by_label.values()))
    l_nc = sum(l_by_label.values())
    u_nc = sum(u_by_label.values())
    yield 'U: %s' % pc(u_nc, n)
    yield 'L: %s' % pc(l_nc, n)


if __name__ == '__main__':
    for line in plac.call(main):
        print line
