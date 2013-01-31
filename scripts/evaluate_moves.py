"""Process a best_moves file to give move P/R/F values"""
import plac
from collections import defaultdict 
import re

punct_re = re.compile(r'-P$')
label_re = re.compile(r'-[A-Za-z]*')
def main(moves_loc):
    # TP, FP, FN
    freqs = defaultdict(lambda: [10e-1000, 10e-1000, 10e-1000])
    total = 0
    bad = 0
    for line in open(moves_loc):
        if line.count('\t') == 0: continue
        try:
            gold, test, _ = line.rstrip().split('\t')
        except:
            print repr(line)
            raise
        if punct_re.search(gold) or punct_re.search(test):
            continue
        total += 1
        if not gold:
            bad += 1
            continue
        gold = label_re.sub('', gold)
        test = label_re.sub('', test)
        gold_moves = gold.split(',')
        # Handle multiple golds by just noting false positive, not false negatives
        if len(gold_moves) > 1:
            if test not in gold_moves:
                freqs[test][1] += 1
            else:
                freqs[test][0] += 1
            continue
        gold = gold_moves[0]
        if test == gold:
            freqs[test][0] += 1
        else:
            freqs[test][1] += 1
            freqs[gold][2] += 1
    print "L\tP\tR\tF"
    for label, (tp, fp, fn) in sorted(freqs.items()):
        p = (float(tp) / (tp + fp + 1e-1000)) if tp + fp > 0 else 0.0
        r = (float(tp) / (tp + fn + 1e-1000)) if tp + fn > 0 else 0.0
        f = (2 * ((p * r) / (p + r + 1e-1000))) if p + r > 0 else 0.0
        print '%s\t%.2f\t%.2f\t%.2f' % (label, p * 100, r * 100, f * 100)
    print '%.2f no good move' % ((float(bad) / total) * 100)
    for repair in ['LU', 'RU', 'RL', 'RR', 'LI']:
        tp, fp, fn = freqs[repair]
        if tp + fp + fn == 0:
            continue
        print '%s %d-%d=%d' % (repair, tp, fp, tp - fp)

if __name__ == '__main__':
    plac.call(main)
        
    
