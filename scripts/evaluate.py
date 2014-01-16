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
        if not sent_str.strip():
            continue
        tokens = [Token(i, tok_str.split()) for i, tok_str in enumerate(sent_str.split('\n'))]
        move_root_deps(tokens)
        flatten_edits(tokens)
        tokens[-1].sbd = True
        for token in tokens:
            yield sent_str, token

def flatten_edits(tokens):
    by_head = defaultdict(list)
    edits = []
    for token in tokens:
        if token.head >= len(tokens):
            token.head = -1
    subtrees = defaultdict(set)
    for token in tokens:
        if token.head > 0 and token.head != token.id:
            subtrees[token.head].add(token)
    edits = [t for t in tokens if t.is_edit or t.label == 'erased']
    visited = set()
    for token in edits:
        if token.id in visited:
            continue
        visited.add(token.id)
        token.label = 'erased'
        token.head = token.id
        token.is_edit = True
        for child in subtrees[token.id]:
            edits.append(child)
    
def move_root_deps(tokens):
    """Deal with unsegmented text by moving all root dependencies to the
    root token, for more stable evaluation."""
    root = len(tokens)
    for token in tokens:
        if token.label.lower() == 'root':
            token.head = root

class Token(object):
    def __init__(self, id_, attrs):
        self.id = id_
        #self.sbd = attrs.pop()
        self.sbd = False
        # CoNLL format
        is_edit = False
        if len(attrs) == 6 or len(attrs) == 5 or len(attrs) == 4:
            attrs.append('False')
            self.dfl_tag = '-'
        elif len(attrs) == 10:
            new_attrs = [str(int(attrs[0]) - 1)]
            new_attrs.append(attrs[1])
            new_attrs.append(attrs[3])
            new_attrs.append(str(int(attrs[6]) - 1))
            dfl_feats = attrs[5].split('|')
            self.dfl_tag = dfl_feats[2]
            new_attrs.append(attrs[7])
            attrs = new_attrs
            attrs.append(str(dfl_feats[2] == '1'))
        self.is_edit = attrs.pop() == 'True'
        self.label = attrs.pop()
        if self.label.lower() == 'root':
            self.label = 'ROOT'
        head = int(attrs.pop())
        self.head = head
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
    u_nc = 0
    l_nc = 0
    ed_tp = 0
    ed_fp = 0
    ed_fn = 0
    ed_n = 0
    rep_tp = 0
    rep_fp = 0
    rep_fn = 0
    rep_n = 0
    tags_corr = 0
    tags_tot = 0
    open_ip = False
    prev_g = None
    prev_t = None
    for (sst, t), (ss, g) in zip(gen_toks(test_loc), gen_toks(gold_loc)):
        tags_corr += t.pos == g.pos
        tags_tot += 1
        if g.label in ["P", 'punct'] and not eval_punct:
            continue
        ed_tp += t.is_edit and g.is_edit
        ed_fp += t.is_edit and not g.is_edit
        ed_fn += g.is_edit and not t.is_edit
        prev_g = g
        prev_t = t
        if g.is_edit:
            ed_n += 1
            continue
        if g.label == 'filler':
            continue
        #if g.dfl_tag != '-': continue
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
    if ed_n != 0:
        ed_p = pc(ed_tp, ed_tp + ed_fp)
        ed_r = pc(ed_tp, ed_n)
        ed_f = 2 * ((ed_p * ed_r) / (ed_p + ed_r + 1e-100))
        yield 'DIS P: %.2f' % ed_p
        yield 'DIS R: %.2f' % ed_r
        yield 'DIS F: %.2f' % ed_f
    yield 'POS Acc: %.2f' % (pc(tags_corr, tags_tot))


if __name__ == '__main__':
    for line in plac.call(main):
        print line
