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


def tokenise_candidate(candidate):
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
            

def read_nbest(nbest_loc, limit=0):
    lines = open(nbest_loc).read().strip().split('\n')
    lines.pop(0)
    if limit:
        lines = lines[:limit]
    for line in lines:
        pieces = line.split()
        _ = pieces.pop(0)
        log_prob = pieces.pop(0)
        words = tokenise_candidate(pieces)
        if words:
            yield float(log_prob), words


def get_nbest(gold_sent, nbest_dir, limit=0):
    turn_id = gold_sent.turn_id
    filename, turn_num = gold_sent.turn_id.split('~')
    speaker = turn_num[0]
    turn_num = turn_num[1:]
    turn_id = '%s%s~%s' % (filename, speaker, turn_num)
    nbest_loc = nbest_dir.join(filename).join(speaker).join('nbest').join(turn_id)
    # Need to copy the gold_sent, as we're going to pass gold_sent to the tagger
    # for training, and the tags get modified by nbest_train.
    gold_copy = Input.from_tokens([(t.word, t.tag, t.head, t.label, t.sent_id, t.is_edit)
                                    for t in gold_sent.tokens])
    if not nbest_loc.exists():
        return [gold_copy]
    gold_tokens = list(gold_sent.tokens)
    gold_sent_id = gold_tokens[0].sent_id
    nbest = []
    seen_gold = False
    for score, candidate in read_nbest(str(nbest_loc), limit=limit):
        cost, edits = get_oracle_alignment(candidate, gold_tokens)
        if cost == 0:
            sent = make_gold_sent(gold_tokens, candidate, edits)
            seen_gold = True
        else:
            sent = make_non_gold_sent(cost, candidate, gold_sent_id)
        nbest.append(sent)
    if not seen_gold:
        nbest.append(gold_copy)
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
            tokens.append(_make_dfl_token(candidate, c_i, sent_id))
            c_i += 1
    return Input.from_tokens(tokens)

def _make_token(word, gold, alignment):
    head = alignment.get(gold.head - 1, -1) + 1
    return (word, gold.tag, head, gold.label,
            gold.sent_id, gold.is_edit)

def _make_dfl_token(words, i, sent_id):
    word = words[i]
    last_word = words[i - 1] if i != 0 else 'EOL'
    next_word = words[i + 1] if i < (len(words) - 1) else 'EOL'
    return (word, 'UH', i + 1, _guess_label(word, last_word, next_word), sent_id, True)


def _guess_label(word, last_word, next_word):
    """
    13117 uh
    7189 you
    7186 know
    4569 well
    3633 oh
    3319 um
    1712 i
    1609 mean
    1050 like
    522 so
    372 huh
    284 now
    213 see
    124 yeah
    108 or
    106 actually
    """
    fillers = set(['uh', 'um', 'uhhuh', 'uh-huh', 'huh', 'oh'])
    discourse = set(['well', 'okay', 'actually', 'like', 'so', 'now', 'yeah'])
    editing = set(['or'])
    if word in fillers:
        return 'fillerF'
    elif word in discourse:
        return 'fillerD'
    elif word in editing:
        return 'fillerE'
    elif word == 'you' and next_word == 'know':
        return 'fillerD'
    elif word == 'know' and last_word == 'you':
        return 'fillerD'
    elif word == 'i' and next_word == 'mean':
        return 'fillerD'
    elif word == 'mean' and last_word == 'i':
        return 'fillerE'
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
    limit=("Limit nbest list to N", "option", "N", int),
    train_tagger=("Train tagger alongside parser", "flag", "p", bool),
    use_edit=("Use the Edit transition", "flag", "e", bool),
    use_break=("Use the Break transition", "flag", "b", bool),
    use_filler=("Use the Filler transition", "flag", "F", bool),
    seed=("Random seed", "option", "s", int)
)
def main(train_loc, nbest_dir, model_loc, n_iter=15,
         feat_set="zhang", feat_thresh=10,
         n_sents=0,
         limit=0,
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
    print "Get sents"
    sents = [Input.from_conll(s) for s in
             train_str.strip().split('\n\n') if s.strip()]
    nbests = [get_nbest(sent, nbest_dir, limit=limit) for sent in sents]
    print "Train"
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
