#!/usr/bin/env python

import plac
from pathlib import Path
import math
import cProfile
import pstats
import sys


import redshift.nbest_parser
from redshift.sentence import Input
from redshift.util import read_nbest, get_nbest_loc
from redshift.util import get_oracle_alignment


def red(string):
    return u'\033[91m%s\033[0m' % string

def green(string):
    return u'\033[92m%s\033[0m' % string

def blue(string):
    return u'\033[94m%s\033[0m' % string

def get_alignment(candidate, gold_words):
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
    return previous_costs[-1], previous_row[-1]



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


def align_to_gold(gold, candidate):

    cost, edits = get_oracle_alignment(candidate, gold)
    tokens = []
    words = list(candidate)
    gold = list(gold)
    g_i = 0
    c_i = 0
    alignment = {}
    for op in edits:
        if op == 'M':
            alignment[g_i] = c_i
            g_i += 1
            c_i += 1
        elif op == 'I' or op == 'fI':
            g_i += 1
        elif op == 'D' or op == 'fD':
            c_i += 1
        else:
            raise StandardError(op)
    g_i = 0
    c_i = 0
    sent_id = gold[0].sent_id
    tokens = []
    for op in edits:
        if op == 'M':
            tokens.append(candidate[c_i])
            g_i += 1
            c_i += 1
        elif op == 'I' or op == 'fI':
            if op == 'fI':
                tokens.append(red('___'))
            g_i += 1
        elif op == 'D' or op == 'fD':
            tokens.append(red(candidate[c_i]))
            c_i += 1
    return ' '.join(tokens)


@plac.annotations(
    mix_weight=("Control the LM and acoustic model mixture", "option", "w", float),
    limit=("Limit N-best to N candidates", "option", "N", int)
)
def main(parser_dir, conll_loc, nbest_dir, mix_weight=0.0, limit=0):
    nbest_dir = Path(nbest_dir)
    #print "Loading parser"
    #parser = redshift.nbest_parser.NBestParser(parser_dir)
    gold_sents = [Input.from_conll(s) for s in
                  open(conll_loc).read().strip().split('\n\n') if s.strip()]
    for gold in gold_sents:
        nbest = read_nbest(get_nbest_loc(gold.turn_id, nbest_dir), limit)
        if not nbest:
            continue
        print align_to_gold(list(gold.tokens), nbest[0][1])
        
        parses = parser.parse_nbest(nbest)
        guess = max(parses, key=lambda sent: sent.score)

        print align_to_gold(list(gold.tokens), [w.word for w in guess if not w.is_edit])
        


if __name__ == "__main__":
    plac.call(main)
