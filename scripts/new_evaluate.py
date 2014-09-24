#!/usr/bin/env python
from __future__ import division

import plac
import json

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
        assert len(gold_tokens) == len(test_tokens), '%d vs %d' % (len(gold_tokens), len(test_tokens))
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
    tp = 1e-100
    fp = 0
    fn = 0
    test_sents = iter(test.sents)
    test_sent = None
    last_t_id = None
    for gold_sent in gold.sents:
        last_g_id = None
        for g in gold_sent:
            if not test_sent:
                test_sent = list(test_sents.next())
                last_t_id = None
            t = test_sent.pop(0)
            if g.is_edit or g.tag == 'UH' or g.label == 'discourse':
                continue
            gold_break = last_g_id is None or g.sent_id != last_g_id
            test_break = last_t_id is None or t.sent_id != last_t_id
            assert g.word == t.word
            last_g_id = g.sent_id
            if not t.is_edit:
                last_t_id = t.sent_id
            tp += test_break and gold_break
            fp += test_break and not gold_break
            fn += gold_break and not test_break
            c += gold_break == test_break
            n += 1
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * p * r) / (p + r)
    print "SBD p: %.2f" % (p * 100)
    print "SBD r: %.2f" % (r * 100)
    print "SBD f: %.2f" % (f * 100)

    return (c / n) * 100 


def main(gold_loc, test_loc, out_loc=None):
    gold = ParsedFile(gold_loc)
    test = ParsedFile(test_loc)
    uas_scorer = Scorer(lambda g, t: g.label == 'P' or g.is_edit or g.label == 'discourse', eval_uas, gold, test)
    las_scorer = Scorer(lambda g, t: g.label == 'P' or g.is_edit or g.label == 'discourse', eval_las, gold, test)
    results = {}
    results['UAS'] = uas_scorer.percent
    results['LAS'] = las_scorer.percent
    p = Scorer(lambda g, t: g.label != 'discourse' and (not t.is_edit or t.label != 'erased'),
               lambda g, t: g.is_edit == t.is_edit, gold, test).percent
    r = Scorer(lambda g, t: g.label != 'discourse' and (not g.is_edit or g.label != 'erased'),
                 lambda g, t: g.is_edit == t.is_edit, gold, test).percent
    sbd_score = score_sbd(gold, test)
    if sbd_score is not None:
        results['SBD'] = sbd_score
    results['DIS P'] = p
    results['DIS R'] = r
    results['DIS F'] = ((2 * p * r) / (p + r))
    if out_loc is not None:
        with open(out_loc, 'w') as file_:
            json.dump(results, file_)
    for result in ['UAS', 'LAS', 'SBD', 'DIS P', 'DIS R', 'DIS F']:
        print '%s: %.3f' % (result, results[result])


if __name__ == '__main__':
    plac.call(main)
