"""Do iterative error-based training

Directory structure assumed:

base_dir/
    gold/
        0/
            train.parses
            train.moves
            held_out
        ...
        train.txt
        devr.txt
        devi.txt
    iters/
        0/
            0/
                train/
                    <to compile>parses
                    <to compile>moves
                eval/
                    <to produce>parses
                    <to produce>moves
                    <to produce>acc
                held_out/
                    <to produce>parses
                    <to produce>moves
                parser/
                    <to produce>model
                    <to produce>features
                    <to produce>words
                    <to produce>pos
"""
from pathlib import Path
import plac
import sh
import time
from math import sqrt
import re
import random

random.seed(0)

def split_data(train_loc, n):
    text = train_loc.open().read().strip()
    if n == 1:
        yield unicode(text), unicode(text)
    else:
        instances = text.split('\n\n')
        length = len(instances)
        test_size = length / n
        train_size = length - test_size
        for i in range(n):
            test_start = i * test_size
            test_end = test_start + test_size
            test = instances[test_start:test_end]
            assert len(test) == test_size
            train = instances[:test_start] + instances[test_end:]
            assert len(train) == train_size
            yield u'\n\n'.join(train), u'\n\n'.join(test)


def setup_base_dir(base_dir, data_dir, train_name, moves_name, n):
    if base_dir.exists():
        sh.rm('-rf', base_dir)
    base_dir.mkdir()
    base_dir.join('iters').mkdir()
    gold_dir = base_dir.join('gold')
    gold_dir.mkdir()
    train_loc = data_dir.join(train_name)
    for i, (train_str, ho_str) in enumerate(split_data(train_loc, n)):
        gold_dir.join(str(i)).mkdir()
        gold_dir.join(str(i)).join('train.parses').open('w').write(train_str)
        gold_dir.join(str(i)).join('held_out').open('w').write(ho_str)
    for i, (train_str, _) in enumerate(split_data(Path(str(train_loc) + '.%s' % moves_name), n)):
        gold_dir.join(str(i)).join('train.moves').open('w').write(train_str)
    gold_dir.join('train.txt').open('w').write(train_loc.open().read())
    gold_dir.join('devr.txt').open('w').write(data_dir.join('devr.txt').open().read())
    gold_dir.join('devi.txt').open('w').write(data_dir.join('devi.txt').open().read())
    gold_dir.join('rgrammar').open('w').write(data_dir.join('rgrammar').open().read())


def setup_fold_dir(base_dir, i, f, z, n_folds, add_gold, parse_percent):
    exp_dir = base_dir.join('iters').join(str(i)).join(str(f))
    if exp_dir.exists():
        sh.rm('-rf', str(exp_dir))
    exp_dir.mkdir(parents=True)
    exp_dir.join('name').open('w').write(u'iter%d_fold%d' % (i, f))
    for name in ['parser', 'train', 'held_out', 'eval']:
        subdir = exp_dir.join(name)
        if not subdir.exists():
            subdir.mkdir()
    train_dir = exp_dir.join('train')
    train_dir.join('rgrammar').open('w').write(base_dir.join('gold').join('rgrammar').open().read())
    parses = train_dir.join('parses').open('w')
    moves = train_dir.join('moves').open('w')
    if i == 0:
        gold_parse_loc = base_dir.join('gold').join(str(f)).join('train.parses')
        parses.write(gold_parse_loc.open().read())
        parses.write(u'\n\n')
    if n_folds == 1:
        exp_dir.join('held_out').join('gold').open('w').write(base_dir.join('gold').join('train.txt').open().read())
    else:
        exp_dir.join('held_out').join('gold').open('w').write(
                base_dir.join('gold').join(str(f)).join('held_out').open().read() + u'\n\n')
    dirs = [d for d in base_dir.join('iters') if int(d.parts[-1]) < i]
    dirs.sort()
    for prev_iter_dir in dirs[z * -1:]:
        folds = list(prev_iter_dir)
        for fold in folds:
            if int(str(fold.parts[-1])) != f or n_folds == 1:
                ho_dir = fold.join('held_out')
                ho_parses = ho_dir.join('gold').open().read().strip().split('\n\n')
                ho_moves = ho_dir.join('moves').open().read().strip().split('\n\n')
                assert len(ho_moves) == len(ho_parses)
                for i in range(len(ho_parses)):
                    if random.uniform(0, 1.0) <= parse_percent:
                        parses.write(ho_parses[i] + u'\n\n')
                        moves.write(ho_moves[i] + u'\n\n')
    parses.close()
    moves.close()
    return exp_dir


def train_and_parse_fold(fold_dir, dev_loc, i, label_set, no_extra_features,
    allow_reattach, allow_unshift, allow_move_top, allow_invert):
    name = fold_dir.join('name').open().read().strip()
    train_args = ['BASE_DIR', 'DEV_LOC', 'LABEL_SET', 'FEAT_STR', 'THRESH',
                   'REPAIR_STR']
    if no_extra_features:
        feat_str = '-x'
    else:
        feat_str = ''
    repair_str = []
    if allow_reattach:
        repair_str.append('-r')
    if allow_move_top:
        repair_str.append('-m')
    if allow_unshift:
        repair_str.append('-u')
    if allow_invert:
        repair_str.append('-v')
    repair_str = ' '.join(repair_str)
    thresh = 5 * i if i >= 1 else 5
    arg_vals = [fold_dir, dev_loc, label_set, feat_str, thresh, repair_str]
    env_str = ','.join('%s=%s' % (k, v) for k, v in zip(train_args, arg_vals))
    sh.qsub('pbs/train.sh', o=fold_dir.join('out'), e=fold_dir.join('err'), v=env_str, N=name)


def check_finished(iter_dir, n):
    finished_jobs = [False for i in range(n)]
    n_done = 0
    for i in range(n):
        exp_dir = iter_dir.join(str(i))
        if exp_dir.join('err').exists():
            finished_jobs[i] = True
            errors = exp_dir.join('err').open().read().strip()
            if errors:
                print errors
                raise StandardError
    return all(finished_jobs)


inst_feats_re = re.compile('(\d+) instances, (\d+) features')
def get_iter_summary(iter_dir, i, n_folds):
    accs = []
    for f in range(n_folds):
        acc = iter_dir.join(str(f)).join('eval').join('acc').open().read()
        uas = [l for l in acc.split('\n') if l.startswith('U')][0].split()[1]
        accs.append(float(uas))
    feats = []
    insts = []
    for f in range(n_folds):
        out_str = iter_dir.join(str(f)).join('out').open().read()
        n_i, n_f = inst_feats_re.search(out_str).groups()
        feats.append(int(n_f))
        insts.append(int(n_i))
    return u'%d    %s    %s    %s' % (i, mean_stdev(accs), mean_stdev(insts, ints=True),
                                mean_stdev(feats, ints=True))


def mean_stdev(nums, ints=False):
    avg = sum(nums) / len(nums)
    var = sum((avg - a)**2 for a in nums) / len(nums)
    stdev = sqrt(var)
    if ints:
        return u'%d (+/- %d)' % (avg, stdev)
    else:
        return u'%.2f (+/- %.2f)' % (avg, stdev)


@plac.annotations(
    n_folds=("Number of splits to use for iterative training", "option", "f", int),
    add_gold=("Always add the gold-standard moves to training", "flag", "g", bool),
    base_dir=("Output directory for model/s", "positional", None, Path),
    data_dir=("Directory of parse data", "positional", None, Path),
    resume_after=("Resume training after N iterations", "option", "s", int),
    no_extra_features=("Don't add extra features", "flag", "x", bool),
    label_set=("Name of label set", "option", "l", str),
    n_iter=("Number of training iterations", "option", "i", int),
    horizon=("How many previous iterations to add", "option", "z", int),
    train_name=("Name of training file", "option", "t"),
    parse_percent=("Percent of held-out parses to use", "option", "p", float),
    allow_reattach=("Allow left-clobber", "flag", "r", bool),
    allow_move_top=("Allow lower/raise of top", "flag", "m", bool),
    allow_unshift=("Allow unshift", "flag", "u", bool),
    allow_invert=("Allow invert", "flag", "v", bool)
)
def main(data_dir, base_dir, n_iter=5, n_folds=5,
         horizon=0, add_gold=False, resume_after=0, no_extra_features=False,
         label_set="MALT", train_name="train.txt", moves_name=None, parse_percent=1.0,
         allow_reattach=False, allow_move_top=False, allow_unshift=False,
         allow_invert=False):
    if moves_name is None:
        if allow_reattach:
            moves_name = 'moves'
        else:
            moves_name = 'moves_base'
    if resume_after <= 0:
        print 'wiping base'
        setup_base_dir(base_dir, data_dir, train_name, moves_name, n_folds)
    log = base_dir.join('log').open('w')
    log.write(u'I\tAcc\tInst.\tFeats.\n')
    print 'Iter  Accuracy           Instances            Features'
    for i in range(resume_after):
        summary = get_iter_summary(base_dir.join('iters').join(str(i)), i, n_folds)
        log.write(summary + u'\n')
        print summary
    if n_folds > 1:
        extra_iters = n_iter
    else:
        extra_iters = 0
    for i in range(resume_after, n_iter + extra_iters):
        if i == n_iter:
            n_folds = 1
        for f in range(n_folds):
            fold_dir = setup_fold_dir(base_dir, i, f, horizon, n_folds, add_gold, parse_percent)
            train_and_parse_fold(fold_dir, data_dir, i, label_set, no_extra_features,
                allow_reattach, allow_unshift, allow_move_top, allow_invert)
        while not check_finished(base_dir.join('iters').join(str(i)), n_folds):
            time.sleep(5)
        summary = get_iter_summary(base_dir.join('iters').join(str(i)), i, n_folds)
        log.write(summary + u'\n')
        print summary
    log.close()

if __name__ == '__main__':
    plac.call(main)
