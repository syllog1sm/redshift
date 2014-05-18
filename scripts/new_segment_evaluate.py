from __future__ import division
import plac

def read_sents(loc):
    sents = open(loc).read().strip().split('\n\n')
    for sent in sents:
        tokens = [t.split() for t in sent.split('\n')]
        yield tokens
        

def main(test_loc, gold_loc):
    test_sents = list(read_sents(test_loc))
    gold_sents = list(read_sents(gold_loc))
    assert len(test_sents) == len(gold_sents)
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    for test, gold in zip(test_sents, gold_sents):
        assert len(test) == len(gold)
        for t, g in zip(test, gold):
            assert t[0] == g[0]
            if g[2] == 'T' and t[2] == 'T':
                tp += 1
            elif g[2] == 'T' and t[2] == 'F':
                fn += 1
            elif g[2] == 'F' and t[2] == 'T':
                fp += 1
            elif g[2] == 'F' and t[2] == 'F':
                tn += 1
            else:
                print g
                print t
                raise StandardError
    print tp
    print fp
    print fn
    print tn
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    f = (2 * p * r) / (p + r)
    print 'P: %.2f' % p
    print 'R: %.2f' % r 
    print 'F: %.2f' % f
    print 'Err: %.4f' % ((fp + fn) / (tp + fp + fn + tn))
    print 'NIST: %.2f' % ((fp + fn) / (tp + fn))
            

if __name__ == '__main__':
    plac.call(main)
