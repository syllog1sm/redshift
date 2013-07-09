"""
Permute dependency files
"""
import plac
import sys
from collections import defaultdict

class Token(object):
    def __init__(self, idx, word, pos):
        self.word = word
        self.pos = pos
        self.idx = int(idx)

class Tree(object):
    def __init__(self, dep_strs):
        self.tokens = []
        heads = []
        self.labels = {}
        for i, line in enumerate(dep_strs.split('\n')):
            if not line.strip():
                continue
            word, pos, head, label = line.split()
            token = Token(i, word, pos)
            self.tokens.append(token)
            self.labels[token] = label
            heads.append((token, int(head)))
        self.heads = {}
        for child, head_idx in heads:
            if head_idx == -1:
                continue
            head = self.tokens[head_idx]
            self.heads[child] = head

    
    def get_head(self):
        for token in self.tokens:
            if token not in self.heads:
                return token

    def swap_dep(self, old_head, old_child, label):
        if old_head in self.heads:
            gp = self.heads[old_head]
            self.heads[old_child] = gp
        else:
            self.heads.pop(old_child)
        self.labels[old_child] = label
        self.heads[old_head] = old_child
        for token in self.tokens:
            if self.heads.get(token) == old_head:
                if old_child > old_head and token > old_child:
                    self.heads[token] = old_child
                if old_child < old_head and token < old_child:
                    self.heads[token] = old_child

    def __str__(self):
        strings = []
        for word in self.tokens:
            if word in self.heads:
                head_idx = str(self.heads[word].idx)
            else:
                head_idx = '-1'
            strings.append('\t'.join((word.word, word.pos, head_idx, self.labels[word])))
        return '\n'.join(strings)


def is_aux(head, child, label):
    if label in ('auxpass', 'aux'):
        return True
    return False


def copula_as_head(tree):
    head = tree.get_head()
    for child in tree.tokens:
        label = tree.labels[child]
        if tree.heads.get(child) == head and is_aux(head, child, label):
            tree.swap_dep(head, child, label)
            tree.labels[head] = tree.labels[child]
            tree.labels[child] = 'ROOT'
            break

def verb_as_head(tree):
    head = tree.get_head()
    for child, label in tree.children[head]:
        if is_aux(child, head, label):
            tree.swap_dep(head, child)
            break

def read_trees(file_):
    lines = file_.read()
    for sent_str in lines.split('\n\n'):
        if not sent_str.strip():
            continue
        yield Tree(sent_str)

def main():
    for tree in read_trees(sys.stdin):
        copula_as_head(tree)
        print str(tree)
        print

if __name__ == '__main__':
    plac.call(main)
