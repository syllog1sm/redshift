from collections import defaultdict

from redshift.sentence cimport Input, Sentence, Step, Token
from index.lexicon cimport Lexeme, DELETED_WORD
from index.lexicon cimport lookup, get_str
from index.hashes import encode_pos, encode_label


def add_gold_parse(Input py_lattice, Input py_gold):
    cdef Sentence* gold = py_gold.c_sent
    cdef Sentence* asr = py_lattice.c_sent
    cdef Lexeme* word
    offset = 0
    asr_to_gold = {}
    gold_to_asr = {}
    for i in range(1, asr.n):
        word = asr.tokens[i].word
        if word == &DELETED_WORD:
            offset += 1
        else:
            assert i-offset not in gold_to_asr
            assert word == gold.tokens[i - offset].word
            asr_to_gold[i] = i - offset
            gold_to_asr[i-offset] = i
    cdef size_t del_tag = encode_pos(b'DEL')
    cdef size_t del_label = encode_label(b'*delete*')
    cdef Token* g
    for i in range(1, asr.n-1):
        if asr.tokens[i].word == &DELETED_WORD:
            asr.tokens[i].is_edit = True
            asr.tokens[i].label = del_label
            asr.tokens[i].head = i
            asr.tokens[i].tag = del_tag
        else:
            g = &gold.tokens[asr_to_gold[i]]
            asr.tokens[i].tag = g.tag
            asr.tokens[i].head = gold_to_asr[g.head]
            asr.tokens[i].label = g.label if not g.is_edit else del_label
            asr.tokens[i].is_edit = g.is_edit
    

def _make_dfl_token(words, i, sent_id):
    word = words[i].word
    last_word = words[i - 1].word if i != 0 else 'EOL'
    next_word = words[i + 1].word if i < (len(words) - 1) else 'EOL'
    return (word, 'UH', i + 1, _guess_label(word, last_word, next_word), sent_id, True)


def _make_token(word, gold, alignment):
    head = alignment.get(gold.head - 1, -1) + 1
    return (word, gold.tag, head, gold.label,
            gold.sent_id, gold.is_edit)


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



def read_lattice(lattice_loc, add_gold=False, limit=0):
    lines = open(lattice_loc).read().strip().split('\n')
    turn_id = lines.pop(0).split()[-1]
    numaligns = lines.pop(0)
    posterior = lines.pop(0)
    lattice = []
    parse = []
    while lines:
        guesses = lines.pop(0).split()[2:]
        ref = lines.pop(0).split()[-1]
        if guesses[0] == '<s>' and len(guesses) == 4:
            continue
        step = []
        idx = 0
        extra_tokens = defaultdict(float)
        extra_tokens['*DELETE*'] = 0.0
        while guesses:
            word = guesses.pop(0)
            prob = float(guesses.pop(0))
            word, retokenized = _adjust_word(word)
            assert word
            extra_tokens[retokenized] += prob
            step.append((prob, word))
        # If we've added duplicate words to the lattice
        # step, do deduplication, where we simply sum the probabilities for
        # each word. Also re-sorts the step, ensuring the right order
        step = _deduplicate_step(step)
        # Apply limit, if any
        if limit and not add_gold:
            step = step[:limit]
        # Find the (retokenized) reference if we're using gold-standard
        ref, extra_ref = _adjust_word(ref)
        idx = [w for p, w in step].index(ref) if add_gold else 0
        lattice.append(step)
        parse.append((idx, None, None, None, None, None))
        # Extra-tokens will always have an entry for *DELETE* --- if that's the
        # only entry, don't add the extra step.
        if len(extra_tokens) >= 2:
            step = _make_new_step(extra_tokens, extra_ref)
            idx = [w for p, w in step].index(extra_ref) if add_gold else 0
            lattice.append(step)
            parse.append((idx, None, None, None, None, None))
    if not lattice:
        return None
    assert len(lattice) == len(parse)
    normed = []
    for step in lattice:
        z = 1 / sum(p for p, w in step)
        normed.append([(p * z, w) for p, w in step])
        assert 0.9 < sum(p for p, w in normed[-1]) < 1.1
    return Input(normed, parse, turn_id=turn_id, wer=0)


def _make_new_step(word_probs, ref):
    if ref != '*DELETE*' and word_probs['*DELETE*'] == 0:
        word_probs.pop('*DELETE*')
    step = [(p, w) for w, p in word_probs.items()]
    step.sort(reverse=True)
    return step


def _deduplicate_step(step):
    word_probs = defaultdict(float)
    for p, w in step:
        word_probs[w] += p
    new_step = [(p, w) for w, p in word_probs.items()]
    new_step.sort(reverse=True)
    return new_step


def _adjust_word(word):
    if word == '[laugh]':
        return '*DELETE*', '*DELETE*'
    if word == 'uhhuh':
        return 'uh-huh', '*DELETE*'
    if "'" not in word:
        return word, '*DELETE*'
    if word[-1] == "'" and word[-2] != "s":
        return word.replace("'", ''), "*DELETE*"
    suffixes = set(["n't", "'s", "'d", "'re", "'ll", "'m", "'ve", "'", "'n",
                    "'cause", "'em", "'ud"])
    for suffix in suffixes:
        if word.endswith(suffix) and word != suffix:
            return word[:-len(suffix)], suffix
    return word, '*DELETE*'


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


