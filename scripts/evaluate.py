#!/usr/bin/env python
import os
import sys
from pathlib import Path
import plac
from collections import defaultdict

def pc(num, den):
    return (num / float(den)) * 100

def fmt_acc(label, n, l_corr, u_corr, total_errs):
    l_pc = pc(l_corr, n)
    u_pc = pc(u_corr, n)
    err_pc = pc(n - l_corr, total_errs)
    return '%s\t%d\t%.1f\t%.1f\t%.1f' % (label, n, l_pc, u_pc, err_pc)


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
    n_by_label = defaultdict(lambda: defaultdict(int))
    u_by_label = defaultdict(lambda: defaultdict(int))
    l_by_label = defaultdict(lambda: defaultdict(int))
    N = 0
    u_nc = 0
    l_nc = 0
    for test_sent, gold_sent in zip(test_sents, gold_sents):
        test_sent = test_sent.split('\n')
        gold_sent = gold_sent.split('\n')
        assert len(test_sent) == len(gold_sent)
        for i, (t, g) in enumerate(zip(test_sent, gold_sent)):
            t = t.strip().split()
            g = g.strip().split()
            g_label = g[-1]
            if g_label == "P" and not eval_punct:
                continue
            t_label = t[-1]
            g_head = g[-2]
            if int(g_head) > i:
                d = 'L'
            else:
                d = 'R'
            t_head = t[-2]
            u_c = g_head == t_head
            l_c = u_c and g_label == t_label
            
            N += 1
            l_nc += l_c
            u_nc += u_c

            n_by_label[d][g_label] += 1
            u_by_label[d][g_label] += u_c
            l_by_label[d][g_label] += l_c
    n_l_err = N - l_nc
    for D in ['L', 'R']:
        yield D 
        n_other = 0
        l_other = 0
        u_other = 0
        for label, n in sorted(n_by_label[D].items(), key=lambda i: i[1], reverse=True):
            if n == 0:
                continue
            elif n < 400:
                n_other += n
                l_other += l_by_label[D][label]
                u_other += u_by_label[D][label]
            else:
                l_corr = l_by_label[D][label]
                u_corr = u_by_label[D][label]
                yield fmt_acc(label, n, l_corr, u_corr, n_l_err)
        yield fmt_acc('Other', n_other, l_other, u_other, n_l_err) 
    yield 'U: %.1f' % pc(u_nc, N)
    yield 'L: %.1f' % pc(l_nc, N)


if __name__ == '__main__':
    for line in plac.call(main):
        print line
