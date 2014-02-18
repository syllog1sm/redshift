from __future__ import division

import plac

from redshift import Sentence, Token

class Scorer(object):
    def __init__(self, skip, comparator, gold, test):
        self.t = 0
        self.f = 0
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
            self.sents.append(Sentence.from_conll(i, sent_str))

    @property
    def tokens(self):
        for sent in self.sents:
            for token in sent.tokens:
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

def is_disfluent(g, t):
    return g.is_edit or g.label in ['erased', 'filler', 'discourse', 'P', 'punct']


def main(gold_loc, test_loc):
    gold = ParsedFile(gold_loc)
    test = ParsedFile(test_loc)
    uas_scorer = Scorer(is_disfluent, eval_uas, gold, test)
    las_scorer = Scorer(is_disfluent, eval_las, gold, test)
    print 'U: %.3f' % uas_scorer.percent
    print 'L: %.3f' % las_scorer.percent
    p = Scorer(lambda g, t: not t.is_edit,
               lambda g, t: g.is_edit == t.is_edit, gold, test).percent
    r = Scorer(lambda g, t: not g.is_edit,
                 lambda g, t: g.is_edit == t.is_edit, gold, test).percent

    print 'DIS P: %.2f' % p
    print 'DIS R: %.2f' % r
    print 'DIS F: %.2f' % ((2 * p * r) / (p + r))


if __name__ == '__main__':
    plac.call(main)
