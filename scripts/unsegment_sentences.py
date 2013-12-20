"""Connect sentences into one big dependency tree."""
import plac
import random
import sys

ROOT_LABEL = 'root'


class Sentence(object):
    def __init__(self, lines):
        self.lines = lines
        self.tokens = [Token(i, line.split()) for i, line in enumerate(lines)]
        self.last_head = -1
        assert self.tokens

    @property
    def head(self):
        for token in self.tokens:
            if token.head == -1:
                return token
        print >> sys.stderr, '\n'.join(self.lines)
        raise StandardError

    def to_str(self, offset, last_head):
        return '\n'.join(token.to_str(offset, last_head) for token in self.tokens)


class Token(object):
    def __init__(self, idx, fields):
        if len(fields) == 10:
            self.dfl = fields[5]
            fields = [fields[1], fields[3], int(fields[6]) - 1, fields[7]]
        else:
            self.dfl = 'A1|-|0|-|-'
        self.idx = idx
        self.word = fields.pop(0)
        self.pos = fields.pop(0)
        self.head = int(fields.pop(0))
        self.label = fields.pop(0)
        assert not fields

    def to_str(self, offset, next_head):
        if self.head == -1:
            if next_head == -1:
                head_idx = -1
            else:
                head_idx = next_head + offset
        else:
            head_idx = offset + self.head
        fields = [str(self.idx + 1), self.word, '-', self.pos, self.pos, self.dfl,
                  str(head_idx + 1), self.label, '-', '-']
        return '\t'.join(fields)


def read_sentences(text):
    sents = text.split('\n\n')
    for sent in sents:
        if not sent.strip():
            continue
        lines = sent.split('\n')
        yield Sentence(lines)


def main(loc):
    with open(loc) as file_:
        text = file_.read()
        last_head = -1
        offset = 0
        sents = list(read_sentences(text))
        i = 0
        sent = sents.pop(0)
        while sents:
            next_sent = sents.pop(0)
            segment_now = sent.head.dfl[:3] != next_sent.head.dfl[:3]
            if segment_now:
                print sent.to_str(offset, -1)
                print
                offset = 0
            else:
                print sent.to_str(offset, next_sent.head.idx + len(sent.tokens))
                offset += len(sent.tokens)
            sent = next_sent


if __name__ == '__main__':
    plac.call(main)
