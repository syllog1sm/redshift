"""See how well the parse scores predict word orderings."""

import random
import sys

import plac

import redshift.parser
from redshift.io_parse import read_pos

@plac.annotations(
    n=("Number of shufflings per sentence", "positional", None, int),
)
def main(n, parser_dir, text_loc=None):
    parser = redshift.parser.load_parser(parser_dir)
    text = open(text_loc) if text_loc is not None else sys.stdin
    w = 0; r = 0
    w_scores = []; r_scores = []
    for sent_str in text:
        versions = [sent_str]
        seen = set()
        seen.add(sent_str.strip())
        tokens = sent_str.split()
        if len(tokens) < 3:
            continue
        for i in range(n):
            random.shuffle(tokens)
            reordering = ' '.join(tokens)
            if reordering not in seen:
                versions.append(reordering)
                seen.add(reordering)
        if len(seen) < 3:
            continue
        parsed = redshift.io_parse.read_pos('\n'.join(versions))
        parser.add_parses(parsed)
        scores = parsed.scores
        idx, score = max(enumerate(scores), key=lambda i_s: i_s[1])
        if idx == 0:
            r += 1
            r_scores.append(score)
        else:
            w += 1
            w_scores.append(score) 
    print r, w, float(r) / r + w
            
        
if __name__ == '__main__':
    plac.call(main)
