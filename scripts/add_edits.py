"""Add edit tokens to a parsed file, so that it aligns with the gold-standard"""
import sys
import plac

def add_toks(test, gold):
    new = []
    if not test or not any(test):
        for gold_idx, g in enumerate(gold):
            new.append('%d\t%s\tNN\t%d\terased' % (gold_idx, g, gold_idx))
        return '\n'.join(new)
    test.reverse()
    gold.reverse()
    test_idx = 0
    id_map = {}
    edits = []
    for gold_idx, g in enumerate(gold):
        if test_idx < len(test):
            t = test[test_idx].split()
        else:
            t = None
        if t is None or g != t[1]:
            edits.append((gold_idx, g))
        else:
            id_map[t[0]] = str(len(gold)- (gold_idx + 1))
            test_idx += 1
    for t in test:
        t = t.split()
        t[0] = id_map[t[0]]
        if t[3] != '-1':
            t[3] = id_map[t[3]]
        new.append('\t'.join(t))
    for idx, word in edits:
        mapped = len(gold) - (idx+1)
        new.insert(idx, '%d\t%s\tNN\t%d\terased' % (mapped, word, mapped))
    return '\n'.join(reversed(new))

def main(test, gold):
    test_sents = open(test).read().strip().split('\n\n')
    gold_sents = open(gold).read().strip().split('\n')
    assert len(test_sents) == len(gold_sents), '%d vs %d' % (len(test_sents), len(gold_sents))
    for test_sent, gold_sent in zip(test_sents, gold_sents):
        if not test_sent.strip() and not gold_sent.strip(): continue
        test_toks = test_sent.split('\n')
        gold_sent = gold_sent.replace('you/PRP know/VBP', 'you_know/MWE').replace('i/PRP mean/VBP', 'i_mean/MWE')
        gold_toks = [g.rsplit('/', 1)[0] for g in gold_sent.split()]
        try:
            test_sent = add_toks(test_toks, gold_toks)
        except:
            print >> sys.stderr, test_sent
            print >> sys.stderr, gold_sent
            raise
        print test_sent
        print

if __name__ == '__main__':
    plac.call(main)
