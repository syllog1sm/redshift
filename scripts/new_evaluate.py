#!/usr/bin/env python
from __future__ import division

import plac

from redshift.sentence import Input

class Scorer(object):
    def __init__(self, skip, comparator, gold, test):
        self.t = 0
        self.f = 0
        self.s = 0
        self.skip = skip
        self.comparator = comparator
        self.eval_file(gold, test)

    def eval_file(self, gold, test):
        gold_tokens = list(gold.tokens)
        test_tokens = list(test.tokens)
        assert len(gold_tokens) == len(test_tokens)
        for g, t in zip(gold_tokens, test_tokens):
            assert g.word == t.word
            if self.skip(g, t):
                self.s += 1
                continue
            if self.comparator(g, t):
                self.t += 1
            else:
                self.f += 1

    @property
    def percent(self):
        return (self.t / (self.t + self.f)) * 100


class ParsedFile(object):
    def __init__(self, loc):
        self.sents = []
        for i, sent_str in enumerate(open(loc).read().strip().split('\n\n')):
            if not sent_str.strip():
                continue
            self.sents.append(list(Input.from_conll(sent_str).tokens))

    @property
    def tokens(self):
        for sent in self.sents:
            for token in sent:
                yield token


def eval_uas(g, t):
    if g.label.lower() == 'root':
        gold_head = 'R'
    else:
        gold_head = g.head - g.id
    if t.label.lower() == 'root':
        test_head = 'R'
    else:
        test_head = t.head - t.id
    return gold_head == test_head


def eval_las(g, t):
    return eval_uas(g, t) and g.label.lower() == t.label.lower()


def score_sbd(gold, test):
    c = 0
    n = 0
    assert len(gold.sents) == len(test.sents)
    for gold_sent, test_sent in zip(gold.sents, test.sents):
        last_g_id = None
        last_t_id = None
        for g, t in zip(gold_sent, test_sent):
            if g.is_edit or g.tag == 'UH':
                continue
            gold_break = last_g_id != None and g.sent_id != last_g_id
            test_break = last_t_id != None and t.sent_id != last_t_id
            assert g.word == t.word
            last_g_id = g.sent_id
            if not t.is_edit and not t.is_filler:
                last_t_id = t.sent_id
            #print g.word, g.tag, gold_break, test_break
            c += gold_break == test_break
            n += 1
    return (c / n) * 100 


def main(gold_loc, test_loc):
    gold = ParsedFile(gold_loc)
    test = ParsedFile(test_loc)
    uas_scorer = Scorer(lambda g, t: g.is_edit, eval_uas, gold, test)
    las_scorer = Scorer(lambda g, t: g.is_edit, eval_las, gold, test)
    print 'U: %.3f' % uas_scorer.percent
    print 'L: %.3f' % las_scorer.percent
    p = Scorer(lambda g, t: not t.is_edit or t.label != 'erased',
               lambda g, t: g.is_edit == t.is_edit, gold, test).percent
    r = Scorer(lambda g, t: not g.is_edit or g.label != 'erased',
                 lambda g, t: g.is_edit == t.is_edit, gold, test).percent
    print 'SBD: %.3f' % score_sbd(gold, test)
    print 'DIS P: %.2f' % p
    print 'DIS R: %.2f' % r
    print 'DIS F: %.2f' % ((2 * p * r) / (p + r))


if __name__ == '__main__':
    plac.call(main)
