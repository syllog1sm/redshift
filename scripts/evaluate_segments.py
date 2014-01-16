"""Give P/R/F of sentence segmentation boundaries."""
from __future__ import division
import plac

class Token(object):
    def __init__(self, line, utterance):
        fields = line.split()
        self.idx = int(fields[0]) - 1
        self.text = fields[1]
        self.pos = fields[3]
        self.dfl = fields[5].split('|')
        self.is_edit = bool(int(self.dfl[2]))
        self.head = int(fields[6]) - 1
        self.label = fields[7]
        anc = self
        utterance.append(self)
        while anc.head < anc.idx:
            if anc.label == 'root' or anc.head == -1:
                self.eol = True
                break
            anc = utterance[anc.head]
        else:
            self.eol = False


def read_sents(conll_loc):
    tokens = []
    utterance = []
    for line in open(conll_loc):
        if not line.strip():
            tokens[-1].eol = True
            utterance = []
            continue
        tokens.append(Token(line, utterance))
    return tokens

def main(test_loc, gold_loc, ignore_edits=True):
    test_tokens = list(read_sents(test_loc))
    gold_tokens = list(read_sents(gold_loc))
    assert len(test_tokens) == len(gold_tokens)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for test, gold in zip(test_tokens, gold_tokens):
        if ignore_edits and gold.is_edit:
            continue
        if gold.eol and test.eol:
            tp += 1
        elif gold.eol:
            fn += 1
        elif test.eol:
            fp += 1
        else:
            tn += 1
    P = tp / (tp + fp)
    R = tn / (tn + fn)
    print 'P: %d/%d=%.3f' % (tp, (tp + fp), P)
    print 'R: %d/%d=%.3f' % (tn, (tn + fn), R)


if __name__ == '__main__':
    plac.call(main)
    
