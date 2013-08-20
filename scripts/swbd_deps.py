"""Edit a SWBD conll format file in various ways"""

import plac
import sys

class Token(object):
    def __init__(self, line):
        props = line.split()
        self.id = int(props[0])
        self.word = props[1]
        self.pos = props[3].split('^')[-1]
        self.label = props[7]
        self.head = int(props[6])
        self.is_edit = props[-1] == 'True'

    def to_str(self):
        props = (self.id, self.word, self.pos, self.pos, self.head,
                 self.label, self.is_edit)
        return '%d\t%s\t-\t%s\t%s\t-\t%d\t%s\t-\t%s' % props


class Sentence(object):
    def __init__(self, sent_str, use_dps):
        self.tokens = [Token(line) for line in sent_str.split('\n')]

    def mark_dps_edits(self):
        edit_depth = 0
        saw_ip = False
        for i, token in enumerate(self.tokens):
            if token.word == r'\]' and saw_ip == 0:
                continue
            if token.word == r'\[':
                edit_depth += 1
                saw_ip = False
            token.is_edit = edit_depth >= 1 or token.is_edit
            if token.word == r'\+':
                edit_depth -= 1
                saw_ip = True
            if token.word == r'\]' and not saw_ip:
                # Assume prev token is actually repair, not reparandum
                # This should only effect 3 cases
                self.tokens[i - 1].is_edit = False
                edit_depth -= 1

    def to_str(self):
        return '\n'.join(token.to_str() for token in self.tokens)

    def label_edits(self):
        """
        Assign the label "erased" to edit tokens headed by non-edit words. Probably
        this should be handled inside the parser instead.
        """
        for i, token in enumerate(self.tokens):
            if token.pos == 'UH':
                continue
            head = self.tokens[token.head - 1]
            if token.is_edit and (head.pos == '-DFL-' or not head.is_edit):
                token.label = 'erased'

    def label_interregna(self):
        for i, token in enumerate(self.tokens):
            if i == 0: continue
            prev = self.tokens[i - 1]
            if (prev.is_edit or prev.label == 'interregnum') \
              and not token.is_edit \
              and token.label in ('discourse', 'parataxis'):
                token.label = 'interregnum'

    def merge_mwe(self, mwe, parent_label=None, new_label=None):
        strings = mwe.split('_')
        assert len(strings) == 2
        for i, token in enumerate(self.tokens):
            if i == 0: continue
            prev = self.tokens[i - 1]
            if prev.word.lower() != strings[0] or token.word.lower() != strings[1]:
                continue
            if token.head == i:
                child = token
                head = prev
            elif prev.head == (i + 1):
                child = prev
                head = token
            else:
                print prev.word, token.word, prev.head, token.head, i
                continue
            if parent_label is not None and head.label != parent_label:
                continue
            head.word = mwe
            head.pos = 'MWE'
            child.word = '<erased>'
            if new_label is not None:
                head.label = new_label
        self.rm_tokens(lambda t: t.word == '<erased>')

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
            try:
                token.head = id_map[token.head]
            except:
                print >> sys.stderr, token.word
                raise
            if token.head > n:
                token.head = 0
            if token.head == token.id:
                token.head -= 1

    def lower_case(self):
        for token in self.tokens:
            token.word = token.word.lower()

@plac.annotations(
    ignore_unfinished=("Ignore unfinished sentences", "flag", "u", bool),
    use_dps=("Use dps in addition to EDITED nodes", "flag", "d", bool),
    merge_mwe=("Merge multi-word expressions", "flag", "m", bool),
    excise_edits=("Clean edits entirely", "flag", "e", bool),
    label_edits=("Label edits", "flag", "l", bool),
    label_interregna=("Label interregna", "flag", "i", bool),
    rm_fillers=("Discard filled pauses", "flag", "f", bool)
)
def main(in_loc, ignore_unfinished=False, use_dps=False, excise_edits=False,
         label_edits=False, merge_mwe=False, label_interregna=False, rm_fillers=False):
    sentences = [Sentence(sent_str, use_dps) for sent_str in
                 open(in_loc).read().strip().split('\n\n')]
    punct = set([',', ':', '.', ';', 'RRB', 'LRB', '``', "''"])
 
    for sent in sentences:
        if ignore_unfinished and sent.tokens[-1].word == 'N_S':
            continue
        orig_str = sent.to_str()
        try:
            if use_dps:
                sent.mark_dps_edits()
            if merge_mwe:
                sent.merge_mwe('you_know')
                sent.merge_mwe('i_mean')
                sent.merge_mwe('right_now')
                sent.merge_mwe('a_while')
                sent.merge_mwe('in_fact')
                sent.merge_mwe('pretty_much')
                sent.merge_mwe('of_course', new_label='discourse')
            if excise_edits:
                sent.rm_tokens(lambda token: token.is_edit)
                sent.rm_tokens(lambda token: token.label == 'discourse')
            if label_edits:
                sent.label_edits()
            if rm_fillers:
                sent.rm_tokens(lambda token: token.pos == 'UH')
            sent.rm_tokens(lambda token: token.word.endswith('-'))
            sent.rm_tokens(lambda token: token.pos in punct)
            sent.rm_tokens(lambda token: token.pos == '-DFL-')
            sent.rm_tokens(lambda token: token.word == 'MUMBLEx')
            if label_interregna:
                sent.label_interregna()
            sent.lower_case()
        except:
            print >> sys.stderr, orig_str
            raise
        if len(sent.tokens) >= 3:
            print sent.to_str()
            print


if __name__ == '__main__':
    plac.call(main)

