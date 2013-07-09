"""
Compile valency statistics
"""
import plac

from collections import defaultdict

def main(loc):
    sents = open(loc).read().strip().split('\n\n')
    sents = [[line.split() for line in sent.split('\n')] for sent in sents]
    lvals = defaultdict(lambda: defaultdict(int))
    rvals = defaultdict(lambda: defaultdict(int))
    plvals = defaultdict(lambda: defaultdict(int))
    prvals = defaultdict(lambda: defaultdict(int))
    roots = defaultdict(int)
    seen_pos = set(['ROOT', 'NONE'])
    for sent in sents:
        rdeps = defaultdict(list)
        for i, (w, p, h, l) in enumerate(sent):
            seen_pos.add(p)
            if i > int(h):
                rdeps[int(h)].append(i)
        for head, children in rdeps.items():
            if head == -1:
                head_pos = 'ROOT'
            else:
                head_pos = sent[head][1]
            sib_pos = 'NONE'
            children.sort()
            for i, child in enumerate(children):
                #rvals[head_pos][(sib_pos, sent[child][1])] += 1
                rvals[head_pos][sent[child][1]] += 1
                sib_pos = sent[child][1]
    seen_pos = list(sorted(seen_pos))
    for head in seen_pos:
        for child in seen_pos:
            print head, child, rvals[head][child]
        #for sib in seen_pos:
        #    for child in seen_pos:
        #        print head, sib, child, rvals[head][(sib, child)] 

if __name__ == '__main__':
    plac.call(main)
            
