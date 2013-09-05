"""
Look for phrases in a CoNLL-formatted file that are analysed uniformly.
"""
import plac
from collections import defaultdict

class Token(object):
    def __init__(self, line):
        props = line.split()
        self.id = int(props[0]) - 1
        self.word = props[1]
        self.pos = props[3].split('^')[-1]
        self.label = props[7]
        self.head = int(props[6]) - 1
        self.is_edit = props[-1] == 'True'

    def to_str(self):
        props = (self.id, self.word, self.pos, self.pos, self.head,
                 self.label, self.is_edit)
        return '%d\t%s\t-\t%s\t%s\t-\t%d\t%s\t-\t%s' % props


class Sentence(object):
    def __init__(self, sent_str):
        self.tokens = [Token(line) for line in sent_str.split('\n')]


def find_bigrams(sentences):
    bigram_freqs = defaultdict(int)
    base_freqs = defaultdict(int)
    for sentence in sentences:
        for w1 in sentence.tokens[:-1]:
            w2 = sentence.tokens[w1.id + 1]
            base_freqs[(w1.word, w2.word)] += 1
            if w1.head == w2.id:
                label = w1.label
                direction = 1
            elif w2.head == w1.id:
                label = w2.label
                direction = -1
            else:
                continue
            bigram_freqs[(w1.word, w2.word, label, direction)] += 1
    return bigram_freqs, base_freqs


def main(in_loc):
    sent_strs = open(in_loc).read().strip().split('\n\n')
    sentences = (Sentence(sent_str) for sent_str in sent_strs)
    as_dep, as_string = find_bigrams(sentences)
    for (w1, w2, label, d), freq in as_dep.items():
        rel_freq = float(freq) / as_string[(w1, w2)]
        if rel_freq > 0.95:
            print freq, '%.3f' % rel_freq, w1, w2, label, d

if __name__ == '__main__':
    plac.call(main)

