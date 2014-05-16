"""Give P/R/F of sentence segmentation boundaries."""
from __future__ import division
import plac

class Token(object):
    def __init__(self, line, utterance):
        fields = line.split()
        self.nl = False
        if len(fields) == 6:
            fields.pop(0)
            self.idx = int(fields[0])
            self.text = fields[1]
            self.pos = fields[2]
            self.head = int(fields[3])
            self.label = fields[4].lower()
            self.is_edit = self.label == 'erased'
        else:
            self.idx = int(fields[0]) - 1
            self.text = fields[1]
            self.pos = fields[3]
            dfl = fields[5].split('|')
            self.is_edit = bool(int(dfl[2]))
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
            tokens[-1].nl = True
            utterance = []
            continue
        tokens.append(Token(line, utterance))
    return tokens

def main(test_loc, gold_loc, ignore_edits=False):
    test_tokens = list(read_sents(test_loc))
    gold_tokens = list(read_sents(gold_loc))
    #assert len(test_tokens) == len(gold_tokens)

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    n = 0
    e = 0

    for test, gold in zip(test_tokens, gold_tokens):
        if ignore_edits and gold.is_edit:
            continue
        if test.nl: continue
        if gold.eol and test.eol:
            tp += 1
        elif gold.eol:
            fn += 1
        elif test.eol:
            fp += 1
        else:
            tn += 1
        n += gold.eol
        e += ((gold.eol or test.eol)  and (gold.eol != test.eol))
    P = tp / (tp + fp)
    R = tn / (tn + fn)
    F = 2 * ((P * R) / (P + R))
    print 'P: %d/%d=%.3f' % (tp, (tp + fp), P)
    print 'R: %d/%d=%.3f' % (tn, (tn + fn), R)
    print 'F: %.3f' % F
    print 'E: %d/%d=%.3f' % (e, n, (e/n))


if __name__ == '__main__':
    plac.call(main)
    
