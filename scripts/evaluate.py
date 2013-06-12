#!/usr/bin/env python
import os
import sys
import plac
from collections import defaultdict

def pc(num, den):
    return (num / float(den+1e-100)) * 100

def fmt_acc(label, n, l_corr, u_corr, total_errs):
    l_pc = pc(l_corr, n)
    u_pc = pc(u_corr, n)
    err_pc = pc(n - l_corr, total_errs)
    return '%s\t%d\t%.3f\t%.3f\t%.3f' % (label, n, l_pc, u_pc, err_pc)


def gen_toks(loc):
    lines = open(str(loc))
    token = None
    i = 0
    for line in lines:
        line = line.strip()
        if not line:
            assert token is not None
            token.append(True)
            yield Token(i, token)
            i = 0
            token = None
        else:
            if token is not None:
                token.append(False)
                yield Token(i, token)
            i += 1
            token = list(line.strip().split())
    if token is not None:
        token.append(False)
        yield Token(i, token)


class Token(object):
    def __init__(self, id_, attrs):
        self.id = id_
        self.sbd = attrs.pop()
        self.label = attrs.pop()
        # Make head an offset from the token id, for sent variation
        head = int(attrs.pop())
        if head == -1 or self.label == 'ROOT':
            self.head = id_
        else:
            self.head = head - id_
        self.pos = attrs.pop()
        self.word = attrs.pop()
        self.dir = 'R' if self.head else 'L'
    

@plac.annotations(
    eval_punct=("Evaluate punct transitions", "flag", "p")
)
def main(test_loc, gold_loc, eval_punct=False):
    if not os.path.exists(test_loc):
        test_loc.mkdir()
    n_by_label = defaultdict(lambda: defaultdict(int))
    u_by_label = defaultdict(lambda: defaultdict(int))
    l_by_label = defaultdict(lambda: defaultdict(int))
    N = 0
    u_nc = 0
    l_nc = 0
    sb_tp = 0
    sb_fp = 0
    sb_fn = 0
    sb_n = 0
    tags_corr = 0
    tags_tot = 0
    for t, g in zip(gen_toks(test_loc), gen_toks(gold_loc)):
        sb_n += g.sbd
        if g.sbd:
            sb_tp += t.sbd
            sb_fn += not t.sbd
        else:
            sb_fp += t.sbd
            if t.sbd:
                print 'SBD Err: ', t.word, g.word
        tags_corr += t.pos == g.pos
        tags_tot += 1
        if g.label == "P" and not eval_punct:
            continue
        elif g.label == 'erased' or g.label == 'discourse':
            continue
        assert t.word == g.word, '%s vs %s' %(t.word, g.word)
        u_c = g.head == t.head
        l_c = u_c and g.label == t.label
        N += 1
        l_nc += l_c
        u_nc += u_c
        n_by_label[g.dir][g.label] += 1
        u_by_label[g.dir][g.label] += u_c
        l_by_label[g.dir][g.label] += l_c
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
    yield 'U: %.3f' % pc(u_nc, N)
    yield 'L: %.3f' % pc(l_nc, N)
    sb_p = pc(sb_tp, sb_tp + sb_fp)
    sb_r = pc(sb_tp, sb_n)
    sb_f = 2 * ((sb_p * sb_r) / (sb_p + sb_r + 1e-100))
    yield 'SBD P: %.2f' % sb_p
    yield 'SBD R: %.2f' % sb_r
    yield 'SBD F: %.2f' % sb_f
    yield 'POS Acc: %.2f' % (pc(tags_corr, tags_tot))


if __name__ == '__main__':
    for line in plac.call(main):
        print line
