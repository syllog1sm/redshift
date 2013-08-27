"""Add edit tokens to a parsed file, so that it aligns with the gold-standard"""

import plac

def add_toks(test, gold):
    test_idx = 0
    id_map = {}
    new = []
    for gold_idx, g in enumerate(gold):
        g = g.split()
        t = test[test_idx].split()
        if g[0] != t[0]:
            new.append((gold_idx, g[0], g[1]))
        else:
            new.append(t)
            id_map[str(test_idx + 1)] = str(gold_idx + 1)
            test_idx += 1
    newer = []
    for t in new:
        if len(t) == 3:
            newer.append('\t'.join((t[0], t[1], t[2], t[0], 'erased')))
        else:
            t[0] = id_map[t[0]]
            if t[3] != '0':
                t[3] = id_map[t[3]]
    return '\n'.join(newer)

def main(test, gold):
    test_sents = open(test).read().split('\n\n')
    gold_sents = open(gold).read().split('\n\n')
    assert len(test_sents) == len(gold_sents)
    for test_sent, gold_sent in zip(test_sents, gold_sents):
        test_toks = test_sent.split('\n')
        gold_toks = gold_sent.split('\n')
        if len(test_toks) == len(gold_toks):
            print test_sent
        else:
            test_sent = add_toks(test_toks, gold_toks)
            print test_sent
        print

if __name__ == '__main__':
    plac.call(main)
