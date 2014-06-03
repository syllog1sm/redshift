#!/usr/bin/env python

import plac
from pathlib import Path

import redshift.parser
from redshift.sentence import Input


def get_oracle_wer(candidate, gold_words):
    fluency = [not t.is_edit for t in gold_words]
    previous_row = []
    for i in range(len(gold_words) + 1):
        # The number of inserts needed to take an empty candidate string up
        # to the gold string at position i. Normally, this would just be "i"
        # -- but inserting disfluent words is free. So, we count the number
        # of fluent words up to position i.
        previous_row.append(['I'] * sum(fluency[:i]))
    for i, cand in enumerate(candidate):
        current_row = [ [] * (i + 1) ]
        for j, gold in enumerate(gold_words):
            delete = previous_row[j + 1]
            if gold.is_edit:
                insert = current_row[j]
            else:
                insert = current_row[j] + ['I']
            insert = current_row[j] + ['I']
            if gold.word != cand:
                subst = previous_row[j] + ['S']
            else:
                subst = previous_row[j]
            best = min((insert, delete, subst), key=lambda hist: len(hist))
            current_row.append(best)
        previous_row = current_row
    return previous_row[-1]


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
    gold_str = ' '.join(t.word for t in gold_sent.tokens)
    for score, candidate in read_nbest(str(nbest_loc)):
        if ' '.join(candidate) == gold_str:
            continue
        edits = get_oracle_wer(candidate, list(gold_sent.tokens))
        #if wer == 0:
        #    sent = make_gold(gold_sent.tokens, candidate)
        #else:
        #    sent = make_non_gold(wer, candidate)
        #nbest.append(sent)
    return nbest


def make_gold(gold, candidate):
    tokens = []
    words = list(candidate)
    gold = list(gold)
    offset = 0
    while words:
        word = words.pop(0)
        if word == gold[0].word or gold[0].is_edit:
            tokens.append((word, gold[0].tag, gold[0].head + offset, gold[0].label,
                           gold[0].sent_id, gold[0].is_edit))
            gold.pop(0)
        else:
            offset += 1
            tokens.append((word, 'UH', len(tokens), _guess_label(word),
                          gold[0].sent_id, True))
    sent = Input.from_tokens(tokens)
    print sent.to_conll()
    return sent
    

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
