#!/usr/bin/env python

import plac
from pathlib import Path

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
        previous_row.append(cell)
        previous_costs.append(_edit_cost(cell))
    for i, cand in enumerate(candidate):
        current_row = [ ['D'] * (i + 1) ]
        current_costs = [0]
        for j, gold in enumerate(gold_words):
            if not gold.is_edit and gold.word == cand:
                subst = previous_row[j] + ['M']
                insert = current_row[j] + ['fI']
                delete = previous_row[j + 1] + ['fD']
                s_cost = previous_costs[j]
                i_cost = current_costs[j] + 1
                d_cost = previous_costs[j + 1] + 1
            else:
                subst = previous_row[j] + ['fS']
                insert = current_row[j] + ['I' if gold.is_edit else 'fI']
                delete = previous_row[j + 1] + ['D']
                s_cost = previous_costs[j] + 1
                i_cost = current_costs[j] + (not gold.is_edit)
                d_cost = previous_costs[j + 1]
            
            #assert s_cost == _edit_cost(subst)
            #assert i_cost == _edit_cost(insert)
            #assert d_cost == _edit_cost(delete)
            move_costs = zip((s_cost, i_cost, d_cost), (subst, insert, delete))
            best_cost, best_hist = min(move_costs)
            current_row.append(best_hist)
            current_costs.append(best_cost)
        previous_row = current_row
        previous_costs = current_costs
    #assert previous_costs[-1] == _edit_cost(previous_row[-1])
    return previous_costs[-1], previous_row[-1]


def _edit_cost(edits):
    return sum(e[0] == 'f' for e in edits)


def tokenise_candidate(candidate):
    suffixes = set(["n't", "'s", "'d", "'re", "'ll", "'m", "'ve", "'"])
    words = []
    for word in candidate:
        if word == 'uhhuh':
            word = 'uh-huh'
        if "'" not in word:
            words.append(word)
            continue
        for suffix in suffixes:
            if word.endswith(suffix):
                words.append(word[:-len(suffix)])
                words.append(suffix)
    return words
            

def read_nbest(nbest_loc):
    lines = open(nbest_loc).read().strip().split('\n')
    lines.pop(0)
    for line in lines:
        pieces = line.split()
        _ = pieces.pop(0)
        log_prob = pieces.pop(0)
        words = tokenise_candidate(pieces)
        if pieces:
            yield float(log_prob), words


def get_nbest(gold_sent, nbest_dir):
    turn_id = gold_sent.turn_id
    filename, turn_num = gold_sent.turn_id.split('~')
    speaker = turn_num[0]
    turn_num = turn_num[1:]
    turn_id = '%s%s~%s' % (filename, speaker, turn_num)
    nbest_loc = nbest_dir.join(filename).join(speaker).join('nbest').join(turn_id)
    nbest = [gold_sent]
    if not nbest_loc.exists():
        return nbest
    gold_tokens = list(gold_sent.tokens)
    gold_sent_id = gold_tokens[0].sent_id
    gold_str = ' '.join(t.word for t in gold_tokens)
    for score, candidate in read_nbest(str(nbest_loc)):
        if ' '.join(candidate) == gold_str:
            continue
        cost, edits = get_oracle_alignment(candidate, gold_tokens)
        if cost == 0:
            sent = make_gold_sent(gold_tokens, candidate, edits)
        else:
            sent = make_non_gold_sent(cost, candidate, gold_sent_id)
        nbest.append(sent)
    return nbest

def make_non_gold_sent(wer, words, sent_id):
    tokens = [(word, None, None, None, sent_id, None) for word in words]
    return Input.from_tokens(tokens, wer=wer)


def make_gold_sent(gold, candidate, edits):
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
        elif op == 'I':
            g_i += 1
        elif op == 'D':
            c_i += 1
        else:
            raise StandardError(op)
    g_i = 0
    c_i = 0
    sent_id = gold[0].sent_id
    tokens = []
    for op in edits:
        if op == 'M':
            tokens.append(_make_token(candidate[c_i], gold[g_i], alignment))
            g_i += 1
            c_i += 1
        elif op == 'I':
            g_i += 1
        elif op == 'D':
            tokens.append(_make_dfl_token(candidate[c_i], c_i, sent_id))
            c_i += 1
    return Input.from_tokens(tokens)

def _make_token(word, gold, alignment):
    head = alignment.get(gold.head - 1, -1) + 1
    return (word, gold.tag, head, gold.label,
            gold.sent_id, gold.is_edit)

def _make_dfl_token(word, i, sent_id):
    return (word, 'UH', i + 1, _guess_label(word), sent_id, True)


def _guess_label(word):
    fillers = set(['uh', 'um', 'uhhuh', 'uh-huh'])
    discourse = set(['you', 'know', 'well', 'okay'])
    if word in fillers:
        return 'fillerF'
    elif word in discourse:
        return 'fillerD'
    else:
        return 'erased'

@plac.annotations(
    train_loc=("Training location", "positional"),
    n_iter=("Number of Perceptron iterations", "option", "i", int),
    feat_thresh=("Feature pruning threshold", "option", "f", int),
    debug=("Set debug flag to True.", "flag", None, bool),
    beam_width=("Beam width", "option", "k", int),
    feat_set=("Name of feat set [zhang, iso, full]", "option", "x", str),
    n_sents=("Number of sentences to train from", "option", "n", int),
    train_tagger=("Train tagger alongside parser", "flag", "p", bool),
    use_edit=("Use the Edit transition", "flag", "e", bool),
    use_break=("Use the Break transition", "flag", "b", bool),
    use_filler=("Use the Filler transition", "flag", "F", bool),
    seed=("Random seed", "option", "s", int)
)
def main(train_loc, nbest_dir, model_loc, n_iter=15,
         feat_set="zhang", feat_thresh=10,
         n_sents=0,
         use_edit=False,
         use_break=False,
         use_filler=False,
         debug=False, seed=0, beam_width=4,
         train_tagger=False):
    nbest_dir = Path(nbest_dir)
    if debug:
        redshift.parser.set_debug(True)
    train_str = open(train_loc).read()

    if n_sents != 0:
        print "Using %d sents for training" % n_sents
        train_str = '\n\n'.join(train_str.split('\n\n')[:n_sents])
    sents = [Input.from_conll(s) for s in
             train_str.strip().split('\n\n') if s.strip()]
    nbests = [get_nbest(sent, nbest_dir) for sent in sents]
    redshift.parser.train_nbest(sents, nbests, model_loc,
        n_iter=n_iter,
        train_tagger=train_tagger,
        beam_width=beam_width,
        feat_set=feat_set,
        feat_thresh=feat_thresh,
        use_edit=use_edit,
        use_break=use_break,
        use_filler=use_filler
    )


if __name__ == "__main__":
    plac.call(main)
