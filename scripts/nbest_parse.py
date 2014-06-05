#!/usr/bin/env python
import os
import os.path
import sys
import plac
import time
import pstats
import cProfile
from pathlib import Path
import math

import redshift.parser
from redshift.sentence import Input

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


def read_nbest(nbest_loc, limit):
    lines = open(nbest_loc).read().strip().split('\n')
    lines.pop(0)
    if limit:
        lines = lines[:limit]
    for line in lines:
        pieces = line.split()
        _ = pieces.pop(0)
        log_prob = pieces.pop(0)
        words = tokenise_candidate(pieces)
        if pieces:
            yield math.exp(float(log_prob)), words


def get_nbest(turn_id, nbest_dir, limit=0):
    filename, turn_num = turn_id.split('~')
    speaker = turn_num[0]
    turn_num = turn_num[1:]
    turn_id = '%s%s~%s' % (filename, speaker, turn_num)
    nbest_loc = nbest_dir.join(filename).join(speaker).join('nbest').join(turn_id)
    if not nbest_loc.exists():
        return []
    else:
        return list(read_nbest(str(nbest_loc), limit))


def tokenise_candidate(candidate):
    suffixes = set(["n't", "'s", "'d", "'re", "'ll", "'m", "'ve", "'"])
    words = []
    for word in candidate:
        if word == 'uhhuh':
            word = 'uh-huh'
        if "'" not in word:
            words.append(word + '/NN')
            continue
        for suffix in suffixes:
            if word.endswith(suffix):
                words.append(word[:-len(suffix)] + '/NN')
                words.append(suffix + '/NN')
                break
        else:
            words.append(word + '/NN')
    return ' '.join(words)


def parse_nbest(parser, candidates, mix_weight):
    sentences = []
    norm = 0.0
    for prob, cand_str in candidates:
        sent = Input.from_pos(cand_str)
        parser.parse(sent)
        norm += sent.score
        sentences.append((prob, sent))
    weighted = []
    for prob, sent in sentences:
        w = (sent.score / norm) + (prob * mix_weight)
        weighted.append((w, sent))
    weighted.sort()
    return sentences[0][1], weighted[-1][1]


def get_lower_bound(nbest, gold):
    candidates = [c.replace('/NN', '').split() for _, c in nbest]
    errors = [get_oracle_alignment(c, gold) for c in candidates]
    return min(errors)[0]


@plac.annotations(
    mix_weight=("Control the LM and acoustic model mixture", "option", "w", float),
    limit=("Limit N-best to N candidates", "option", "N", int)
)
def main(parser_dir, conll_loc, nbest_dir, out_dir, mix_weight=0.0, limit=0):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    nbest_dir = Path(nbest_dir)
    print "Loading parser"
    parser = redshift.parser.Parser(parser_dir)
    gold_sents = [Input.from_conll(s) for s in
                  open(conll_loc).read().strip().split('\n\n') if s.strip()]
    wer = 0
    baseline = 0
    oracle = 0
    n = 0
    for gold in gold_sents:
        nbest = get_nbest(gold.turn_id, nbest_dir, limit=limit)
        if not nbest:
            continue
        oracle += get_lower_bound(nbest, list(gold.tokens))
        first, guess = parse_nbest(parser, nbest, mix_weight)
        fluent_gold = [t.word for t in gold.tokens if not t.is_edit]
        fluent_guess = [t.word for t in guess.tokens if not t.is_edit]
        fluent_bl = [t.word for t in first.tokens if not t.is_edit]
        baseline += levenshtein(fluent_gold, fluent_bl)
        wer += levenshtein(fluent_gold, fluent_guess)
        n += len(fluent_gold)
    print wer, n, float(wer) / n
    print baseline, n, float(baseline) / n
    print oracle, n, float(oracle) / n


if __name__ == '__main__':
    plac.call(main)
