"""Read a parse and gold file, and print disfluency errors"""

import plac


class Token(object):
    def __init__(self, id_, attrs):
        self.id = id_
        #self.sbd = attrs.pop()
        self.sbd = False
        # CoNLL format
        is_edit = False
        if len(attrs) == 5 or len(attrs) == 4:
            attrs.append('False')
        elif len(attrs) == 10:
            new_attrs = [str(int(attrs[0]) - 1)]
            new_attrs.append(attrs[1])
            new_attrs.append(attrs[3])
            new_attrs.append(str(int(attrs[6]) - 1))
            new_attrs.append(attrs[7])
            #new_attrs.append(attrs[9])
            fields = attrs[5].split('|')
            new_attrs.append(fields[2] == '1')
            attrs = new_attrs
        self.is_edit = str(attrs.pop()) == 'True'
        self.label = attrs.pop()
        if self.label == 'erased':
            self.is_edit = True
        if self.label.lower() == 'root':
            self.label = 'ROOT'
        head = int(attrs.pop())
        self.head = head
        # Make head an offset from the token id, for sent variation
        #if head == -1 or self.label.upper() == 'ROOT':
        #    self.head = id_
        #else:
        #    self.head = head - id_
        self.pos = attrs.pop()
        self.word = attrs.pop()
        self.dir = 'R' if head >= 0 and head < self.id else 'L'

def get_sents(loc):
    sent_strs = open(str(loc)).read().strip().split('\n\n')
    token = None
    i = 0
    for sent_str in sent_strs:
        yield [Token(i, tok_str.split()) for i, tok_str in enumerate(sent_str.split('\n'))]

def red(string):
    return u'\033[91m%s\033[0m' % string

def green(string):
    return u'\033[92m%s\033[0m' % string

def blue(string):
    return u'\033[94m%s\033[0m' % string


#OKBLUE = '\033[94m'
#OKGREEN = '\033[92m'
#WARNING = '\033[93m'
#FAIL = '\033[91m'
#ENDC = '\033[0m'

def main(parse_loc, gold_loc):
    for parse_sent, gold_sent in zip(get_sents(parse_loc), get_sents(gold_loc)):
        tok_strs = []
        for ptok, gtok in zip(parse_sent, gold_sent):
            if ptok.pos != 'UH' and ptok.is_edit and not gtok.is_edit:
                tok_strs.append(red(ptok.word))
            elif gtok.is_edit and ptok.is_edit:
                tok_strs.append(green(ptok.word))
            elif gtok.is_edit and not ptok.is_edit:
                tok_strs.append(blue(ptok.word))
            else:
                tok_strs.append(ptok.word)
        print ' '.join(tok_strs)

if __name__ == '__main__':
    plac.call(main)
