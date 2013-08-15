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
    sent_strs = open(str(loc)).read().strip().split('\n\n')
    token = None
    i = 0
    for sent_str in sent_strs:
        tokens = [Token(i, tok_str.split()) for i, tok_str in enumerate(sent_str.split('\n'))]
        for token in tokens:
            yield sent_str, token


class Token(object):
    def __init__(self, id_, attrs):
        self.id = id_
        # CoNLL format
        if len(attrs) == 10:
            new_attrs = [str(int(attrs[0]) - 1)]
            new_attrs.append(attrs[1])
            new_attrs.append(attrs[3])
            new_attrs.append(str(int(attrs[6]) - 1))
            new_attrs.append(attrs[7])
            attrs = new_attrs
        self.label = attrs.pop()
        if self.label.lower() == 'root':
            self.label = 'ROOT'
        try:
            head = int(attrs.pop())
        except:
            print orig
            raise
        self.head = head
        # Make head an offset from the token id, for sent variation
        #if head == -1 or self.label.upper() == 'ROOT':
        #    self.head = id_
        #else:
        #    self.head = head - id_
        self.pos = attrs.pop()
        self.word = attrs.pop()
        self.dir = 'R' if head >= 0 and head < self.id else 'L'
    

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
    N_p = 0
    u_nc = 0
    l_nc = 0
    p_nc = 0
    for (sst, t), (ss, g) in zip(gen_toks(test_loc), gen_toks(gold_loc)):
        p_nc += t.pos == g.pos
        N_p += 1
        if g.label in ["P", 'punct'] and not eval_punct:
            continue
        prev_g = g
        prev_t = t
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
            elif n < 100:
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
    yield 'P: %.3f' % pc(p_nc, N_p)

if __name__ == '__main__':
    for line in plac.call(main):
        print line
