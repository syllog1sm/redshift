"""Edit a SWBD conll format file in various ways"""

import plac

class Token(object):
    def __init__(self, line):
        props = line.split()
        self.id = int(props[0])
        self.word = props[1]
        self.pos = props[3].split('^')[-1]
        self.label = props[7]
        self.head = int(props[6])
        self.is_edit = False

    def to_str(self):
        props = (self.id, self.word, self.pos, self.pos, self.head,
                 self.label, self.is_edit)
        return '%d\t%s\t-\t%s\t%s\t-\t%d\t%s\t-\t%s' % props


class Sentence(object):
    def __init__(self, sent_str):
        self.tokens = [Token(line) for line in sent_str.split('\n')]
        edit_depth = 0
        saw_ip = False
        for i, token in enumerate(self.tokens):
            if token.word == r'\]' and saw_ip == 0:
                continue
            if token.word == r'\[':
                edit_depth += 1
                saw_ip = False
            token.is_edit = edit_depth >= 1
            if token.word == r'\+':
                edit_depth -= 1
                saw_ip = True
            if token.word == r'\]' and not saw_ip:
                # Assume prev token is actually repair, not reparandum
                # This should only effect 3 cases
                self.tokens[i - 1].is_edit = False
                edit_depth -= 1
        n_erased = 0
        self.n_dfl = 0
        for token in self.tokens:
            if token.word == r'\[':
                self.n_dfl += 1

    def to_str(self):
        return '\n'.join(token.to_str() for token in self.tokens)

    def label_edits(self):
        for token in self.tokens:
            if token.pos == 'UH':
                continue
            head = self.tokens[token.head - 1]
            if token.is_edit and (head.pos == '-DFL-' or not head.is_edit):
                token.label = 'erased'

    def rm_tokens(self, rejector):
        # 0 is root in conll format
        id_map = {0: 0}
        rejected = set()
        new_id = 1
        for token in self.tokens:
            id_map[token.id] = new_id
            if not rejector(token):
                new_id += 1
            else:
                rejected.add(token.id)
        for token in self.tokens:
            while token.head in rejected:
                head = self.tokens[token.head - 1]
                token.head = head.head
                token = head
        self.tokens = [token for token in self.tokens if not rejector(token)]
        n = len(self.tokens)
        for token in self.tokens:
            token.id = id_map[token.id]
            token.head = id_map[token.head]
            if token.head > n:
                token.head = 0
            if token.head == token.id:
                token.head -= 1

    def lower_case(self):
        for token in self.tokens:
            token.word = token.word.lower()

@plac.annotations(
    ignore_unfinished=("Ignore unfinished sentences", "flag", "u", bool),
    excise_edits=("Clean edits entirely", "flag", "e", bool),
)
def main(in_loc, ignore_unfinished=False, excise_edits=False):
    sentences = [Sentence(sent_str) for sent_str in
                 open(in_loc).read().strip().split('\n\n')]
    punct = set([',', ':', '.', ';', 'RRB', 'LRB', '``', "''"])
 
    for sent in sentences:
        if ignore_unfinished and sent.tokens[-1].word == 'N_S':
            continue
        orig_str = sent.to_str()
        try:
            if excise_edits:
                sent.rm_tokens(lambda token: token.is_edit)
                sent.rm_tokens(lambda token: token.label == 'discourse')
            sent.rm_tokens(lambda token: token.pos == '-DFL-')
            sent.rm_tokens(lambda token: token.pos in punct)
            sent.rm_tokens(lambda token: token.word.endswith('-'))
            sent.rm_tokens(lambda token: token.word == 'MUMBLEx')
            sent.lower_case()
        except:
            print orig_str
            raise
        if len(sent.tokens) >= 3:
            print sent.to_str()
            print


if __name__ == '__main__':
    plac.call(main)

