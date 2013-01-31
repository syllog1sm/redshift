"""Write gold moves file"""

from redshift.parser import Parser
from redshift.io_parse import read_conll

import plac
from pathlib import Path

@plac.annotations(
    label_set=("Label set to use", "option", "l"),
    allow_reattach=("Allow reattach repair", "flag", "r"),
    allow_moves=("Allow right-lower", "flag", "m")
)
def main(train_loc, out_loc, label_set="MALT", allow_reattach=False, allow_moves=False):
    parser_dir = Path('/tmp').join('parser')
    if not parser_dir.exists():
        parser_dir.mkdir()
    grammar_loc = Path(train_loc).parent().join('rgrammar') if allow_reattach else None
    parser = Parser(str(parser_dir), clean=True, label_set=label_set,
                    allow_reattach=allow_reattach, allow_move=allow_moves,
                    grammar_loc=grammar_loc)
    train = read_conll(open(train_loc).read())
    parser.add_gold_moves(train)
    with open(out_loc, 'w') as out_file:
        train.write_moves(out_file)

if __name__ == '__main__':
    plac.call(main)
