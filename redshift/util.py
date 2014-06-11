from os import path
import json
from pathlib import Path
import math


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def write(cls, model_dir, name, **kwargs):
        open(path.join(model_dir, '%s.json' % name), 'w').write(json.dumps(kwargs))

    @classmethod
    def read(cls, model_dir, name):
        return cls(**json.load(open(path.join(model_dir, '%s.json' % name))))


def split_apostrophes(candidate):
    suffixes = set(["n't", "'s", "'d", "'re", "'ll", "'m", "'ve", "'"])
    words = []
    for word in candidate:
        if word == '[laugh]':
            continue
        if word == 'uhhuh':
            word = 'uh-huh'
        if "'" not in word:
            words.append(word)
            continue
        for suffix in suffixes:
            if word.endswith(suffix):
                words.append(word[:-len(suffix)])
                words.append(suffix)
                break
        else:
            words.append(word)
    return words
 

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
 
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = xrange(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]


def get_oracle_alignment(candidate, gold_words):
    # Levenshtein distance, except we need the history, and some operations
    # are zero-cost. Specifically, inserting a word that's disfluent in the
    # gold is 0-cost (why insert it?), and we can always delete additional
    # candidate words (via Edit). Mark costly operations with the string 'f',
    # and score the history using _edit_cost.
    previous_row = []
    previous_costs = []
    for i in range(len(gold_words) + 1):
        cell = []
        for j in range(i):
            cell.append('I' if gold_words[j].is_edit else 'fI')
        previous_row.append(''.join(cell))
        previous_costs.append(_edit_cost(''.join(cell)))
    for i, cand in enumerate(candidate):
        current_row = ['D' * (i + 1) ]
        current_costs = [0]
        for j, gold in enumerate(gold_words):
            if not gold.is_edit and gold.word == cand:
                subst = previous_row[j] + 'M'
                insert = current_row[j] + 'fI'
                delete = previous_row[j + 1] + 'fD'
                s_cost = previous_costs[j]
                i_cost = current_costs[j] + 1
                d_cost = previous_costs[j + 1] + 1
            else:
                subst = previous_row[j] + 'fS'
                insert = current_row[j] + ('I' if gold.is_edit else 'fI')
                delete = previous_row[j + 1] + 'D'
                s_cost = previous_costs[j] + 1
                i_cost = current_costs[j] + (not gold.is_edit)
                d_cost = previous_costs[j + 1]
            
            #assert s_cost == _edit_cost(subst)
            #assert i_cost == _edit_cost(insert)
            #assert d_cost == _edit_cost(delete), 'Cost: %d, string: %s' % (d_cost, delete)
            move_costs = zip((s_cost, i_cost, d_cost), (subst, insert, delete))
            best_cost, best_hist = min(move_costs)
            current_row.append(best_hist)
            current_costs.append(best_cost)
        previous_row = current_row
        previous_costs = current_costs
    #assert previous_costs[-1] == _edit_cost(previous_row[-1])
    return previous_costs[-1], previous_row[-1].replace('f', '')


def _edit_cost(edits):
    return edits.count('f')


def get_nbest_loc(turn_id, nbest_dir):
    nbest_dir = Path(nbest_dir)
    filename, turn_num = turn_id.split('~')
    speaker = turn_num[0]
    turn_num = turn_num[1:]
    turn_id = '%s%s~%s' % (filename, speaker, turn_num)
    nbest_loc = nbest_dir.join(filename).join(speaker).join('nbest').join(turn_id)
    return str(nbest_loc) if nbest_loc.exists() else None


def read_nbest(nbest_loc, limit):
    if nbest_loc is None:
        return []
    lines = open(nbest_loc).read().strip().split('\n')
    lines.pop(0)
    nbest = []
    for line in lines:
        pieces = line.split()
        _ = pieces.pop(0)
        log_prob = pieces.pop(0)
        words = split_apostrophes(pieces)
        if words:
            nbest.append((math.exp(float(log_prob)), words))
    if limit:
        nbest = nbest[:limit]
    return nbest
