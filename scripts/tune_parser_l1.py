"""
Tune the L1 regularisation parameter for the parser
"""
import plac
from pathlib import Path
import redshift.io_parse

import tagging.optimise
import redshift.parser

def get_pos(conll_str):
    pos_sents = []
    for sent_str in conll_str.strip().split('\n\n'):
        sent = []
        for line in sent_str.split('\n'):
            pieces = line.split()
            if len(pieces) == 5:
                pieces.pop(0)
            word = pieces[0]
            pos = pieces[1]
            sent.append('%s/%s' % (word, pos))
        pos_sents.append(' '.join(sent))
    return '\n'.join(pos_sents)


def make_evaluator(parser_dir, solver_type, train_loc, dev_loc):
    def wrapped(l1):
        parser = redshift.parser.Parser(parser_dir, solver_type=solver_type,
                                        clean=True, C=l1)
        dev_gold = redshift.io_parse.read_conll(dev_loc.open().read())
        train = redshift.io_parse.read_conll(train_loc.open().read())
        parser.train(train)
        dev = redshift.io_parse.read_pos(get_pos(dev_loc.open().read()))
        acc = parser.add_parses(dev, gold=dev_gold) * 100
        wrapped.models[l1] = acc
        return acc
    models = {}
    wrapped.models = models
    return wrapped
        

@plac.annotations(
    first_val=("Lower bound", "option", "l", float),
    last_val=("Upper bound", "option", "u", float),
    initial_results=("Initialise results with these known values", "option", "r", str),
    solver_type=("LibLinear solver. Integers following the LibLinear CL args", "option", "s", int)
)
def main(parser_dir, train_loc, dev_loc, solver_type=None, first_val=None, last_val=None, initial_results=None):
    train_loc = Path(train_loc)
    dev_loc = Path(dev_loc)
    learner = make_evaluator(parser_dir, solver_type, train_loc, dev_loc)
    results = []
    if initial_results is not None:
        for res_str in initial_results.split('_'):
            v, s = res_str.split(',')
            results.append((float(v), float(s)))
    if first_val is not None and first_val not in [r[0] for r in results]:
        results.append((first_val, learner(first_val)))
    if last_val is not None and last_val not in [r[0] for r in results]:
        results.append((last_val, learner(last_val)))
    results.sort(key=lambda i: i[0])
    if len(results) == 2:
        mid_point = (results[0][0] + results[-1][0]) / 2
        results.insert(1, (mid_point, learner(mid_point)))
    best_value, best_score  = tagging.optimise.search(learner, results)
    print best_value
    print best_score
 

if __name__ == '__main__':
    plac.call(main)
