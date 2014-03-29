from fabric.api import local, run, lcd, cd, env
from fabric.operations import get, put
from fabric.contrib.files import exists
from pathlib import Path
import time
import re
from math import sqrt
from os.path import join as pjoin
from os import listdir
from StringIO import StringIO
import scipy.stats

from itertools import combinations

env.use_ssh_config = True

from _paths import REMOTE_REPO, REMOTE_CONLL, REMOTE_MALT, REMOTE_STANFORD, REMOTE_PARSERS
from _paths import REMOTE_SWBD, REMOTE_UNSEG_SWBD
from _paths import LOCAL_REPO, LOCAL_MALT, LOCAL_STANFORD, LOCAL_PARSERS
from _paths import HOSTS, GATEWAY

env.hosts = HOSTS
env.gateway = GATEWAY


def recompile(runner=local):
    clean()
    make()

def clean():
    with lcd(str(LOCAL_REPO)):
        local('python setup.py clean --all')

def make():
    with lcd(str(LOCAL_REPO)):
        local('python setup.py build_ext --inplace')

def qstat():
    run("qstat -na | grep mhonn")


def deploy():
    clean()
    make()
    with cd(str(REMOTE_REPO)):
        run('git pull')


def beam(name, k=8, n=1, size=0, train_alg="static", feats="zhang", tb='wsj',
         unlabelled=False, auto_pos='False', iters=15, repairs='False', 
         use_sbd='True'):
    size = int(size)
    k = int(k)
    n = int(n)
    iters = int(iters)
    unlabelled = unlabelled and unlabelled != 'False'
    auto_pos = auto_pos and auto_pos != 'False'
    repairs = repairs and repairs != 'False'
    use_edit = False
    use_sbd = use_sbd == 'True'
    if tb == 'wsj':
        data = str(REMOTE_STANFORD)
        train_name = 'train.txt'
        eval_pos = 'devi.txt'
        eval_parse = 'devr.txt'
    elif tb == 'swbd' or tb == 'unseg_swbd':
        if tb == 'swbd':
            data = str(REMOTE_SWBD)
        elif tb == 'unseg_swbd':
            data = str(REMOTE_UNSEG_SWBD)
        #data = str(REMOTE_SWBD) if tb == 'swbd' else str(REMOTE_UNSEG_SWBD)
        train_name = 'train.conll'
        eval_pos = 'dev.pos'
        eval_parse = 'dev.conll'
        if train_alg == 'dynedit':
            use_edit = True
            train_alg = 'dyn'
    elif tb == 'clean_swbd':
        data = str(REMOTE_SWBD)
        train_name = 'train.clean.conll'
        eval_pos = 'dev.clean.pos'
        eval_parse = 'dev.clean.conll'
    exp_dir = str(REMOTE_PARSERS)
    train_n(n, name, exp_dir,
            data, k=k, i=iters, f=10, feat_str=feats, 
            n_sents=size, train_name=train_name, train_alg=train_alg,
            unlabelled=unlabelled, auto_pos=auto_pos, repairs=repairs,
            use_edit=use_edit, use_sbd=use_sbd, dev_names=(eval_pos, eval_parse))


def qdisfl(size='1000'):
    train_str = r'./scripts/train.py -e -n {size} -a dyn -x disfl -k 8 -p {train} {model}'
    parse_str = r'./scripts/parse.py {model} {pos} {out}'
    eval_str = r'./scripts/evaluate.py {out}/parses {gold} | tee {out}/acc'

    train = '~/data/tacl13_swbd/unseg/train.conll'
    pos = '~/data/tacl13_swbd/unseg/dev.pos'
    gold = '~/data/tacl13_swbd/unseg/dev.conll'
    model = '~/data/parsers/tmp'
    out = '~/data/parsers/tmp/dev'
    with cd(str(REMOTE_REPO)):
        run(train_str.format(size=size, train=train, model=model))
        run(parse_str.format(model=model, pos=pos, out=out))
        run(eval_str.format(out=out, gold=gold))
    

def train_n(n, name, exp_dir, data, k=1, feat_str="zhang", i=15, upd='max',
            train_alg="online", n_sents=0, static=False, use_edit=False,
            use_sbd=True, repairs=False,
            unlabelled=False, ngrams='', t=0, f=0, train_name='train.txt',
            dev_names=('devi.txt', 'devr.txt'), auto_pos=False):
    exp_dir = str(exp_dir)
    repo = str(REMOTE_REPO)
    for seed in range(n):
        exp_name = '%s_%d' % (name, seed)
        model = pjoin(exp_dir, name, str(seed))
        run("mkdir -p %s" % model, quiet=True)
        train_str = _train(pjoin(data, train_name), model, k=k, i=i,
                           feat_str=feat_str, train_alg=train_alg, seed=seed,
                           n_sents=n_sents, use_edit=use_edit, use_sbd=use_sbd,
                           unlabelled=unlabelled,
                           allow_reattach=repairs, allow_reduce=repairs,
                           vocab_thresh=t, feat_thresh=f, auto_pos=auto_pos)
        parse_str = _parse(model, pjoin(data, dev_names[0]), pjoin(model, 'dev'))
        eval_str = _evaluate(pjoin(model, 'dev', 'parses'), pjoin(data, dev_names[1]))
        grep_str = "grep 'U:' %s >> %s" % (pjoin(model, 'dev', 'acc'),
                                           pjoin(model, 'dev', 'uas')) 
        # Save disk space by removing models
        #del_str = "rm %s %s" % (pjoin(model, "model"), pjoin(model, "words"))
        del_str = ''
        script = _pbsify(repo, (train_str, parse_str, eval_str, grep_str, del_str))
        script_loc = pjoin(repo, 'pbs', exp_name)
        with cd(repo):
            put(StringIO(script), script_loc)
            err_loc = pjoin(model, 'stderr')
            out_loc = pjoin(model, 'stdout')
            run('qsub -N %s %s -e %s -o %s' % (exp_name, script_loc, err_loc, out_loc), quiet=True)

def parse_n(name, devname):
    data = str(REMOTE_SWBD)
    exp_dir = str(REMOTE_PARSERS)
    repo = str(REMOTE_REPO)
    #pos = devname + '.pos'
    gold = devname + '.conll'
    pos = '/home/mhonniba/data/swbd_stanford/raw_wazoo_test.pos'
    n = len(run("ls %s" % pjoin(exp_dir, name), quiet=True).split())
    script = []
    for seed in range(n):
        model = pjoin(exp_dir, name, str(seed))
        script.append("mkdir %s" % pjoin(model, devname))
        script.append(_parse(model, pos, pjoin(model, devname)))
        script.append(_add_edits(pjoin(model, devname), pjoin(data, 'test.pos')))
        script.append(_evaluate(pjoin(model, devname, 'pipe.parses'), pjoin(data, gold)))
        script.append("grep 'U:' %s >> %s" % (pjoin(model, devname, 'acc'),
                                           pjoin(model, devname, 'uas')))
    script = _pbsify(repo, script)
    script_loc = pjoin(repo, 'pbs', 'parse_' + name)
    with cd(repo):
        put(StringIO(script), script_loc)
        run('qsub -N %s %s' % ('parse_' + name, script_loc), quiet=True)


def tabulate(prefix, names, terms):
    terms = terms.split('-')
    names = names.split('-')
    rows = [terms]
    print prefix, '&\t',
    print '\t&\t'.join(terms),
    print r'\\'
    for name in names:
        exp_dir = str(REMOTE_PARSERS.join(prefix + '_' + name))
        row = []
        for term in terms:
            results = get_accs(exp_dir, term=term)
            row.append(sum(results) / len(results))
        print name, '&\t',
        print '\t&\t'.join('%.1f' % r for r in row),
        print r'\\'
    

def count_finished(exp_dir):
    with cd(exp_dir):
        samples = [s for s in run("ls %s/*/" % exp_dir, quiet=True).split()
                   if s.endswith('stdout')]
    return len(samples)


def get_accs(exp_dir, eval_name='dev', term='U'):
    results = []
    with cd(exp_dir):
        results = [float(s.split()[-1]) for s in
                run("grep '%s:' %s/*/dev/acc" % (term, exp_dir), quiet=True).split('\n')
                   if s.strip()]
    return results


def _train(data, model, debug=False, k=1, feat_str='zhang', i=15,
           train_alg="static", seed=0, args='',
           n_sents=0, ngrams=0, vocab_thresh=0, feat_thresh=10,
           use_edit=False, use_sbd=True, unlabelled=False, auto_pos=False,
           allow_reattach=False, allow_reduce=False):
    use_edit = '-e' if use_edit else ''
    use_sbd = '-b' if use_sbd else ''
    unlabelled = '-u' if unlabelled else ''
    auto_pos = '-p' if auto_pos else ''
    repairs = '-r' if allow_reattach else ''
    repairs += ' -d' if allow_reduce else ''
    template = './scripts/train.py -i {i} -a {alg} -k {k} -x {feat_str} {data} {model} -s {seed} -n {n_sents} -t {vocab_thresh} -f {feat_thresh} {use_sbd} {use_edit} {repairs} {unlabelled} {auto_pos} {args}'
    if debug:
        template += ' -debug'
    return template.format(data=data, model=model, k=k, feat_str=feat_str, i=i,
                           vocab_thresh=vocab_thresh, feat_thresh=feat_thresh,
                           alg=train_alg, use_edit=use_edit, use_sbd=use_sbd,
                           seed=seed, repairs=repairs,
                          args=args, n_sents=n_sents, ngrams=ngrams,
                          unlabelled=unlabelled, auto_pos=auto_pos)


def _parse(model, data, out, gold=False):
    template = './scripts/parse.py {model} {data} {out} '
    if gold:
        template += '-g'
    return template.format(model=model, data=data, out=out)


def _evaluate(test, gold):
    return './scripts/evaluate.py %s %s > %s' % (test, gold, test.replace('parses', 'acc'))

def _add_edits(test_dir, pos):
    in_loc = pjoin(test_dir, 'parses')
    out_loc = pjoin(test_dir, 'pipe.parses')
    return 'python scripts/add_edits.py %s %s > %s' % (in_loc, pos, out_loc)


def _pbsify(repo, command_strs, size=6):
    header = """#! /bin/bash
#PBS -l walltime=20:00:00,mem=4gb,nodes=1:ppn={n_procs}
source /home/mhonniba/ev/bin/activate
export PYTHONPATH={repo}:{repo}/redshift:{repo}/svm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/lib64:/usr/lib64/:/usr/lib64/atlas:{repo}/redshift/svm/lib/
cd {repo}"""
    return header.format(n_procs=size, repo=repo) + '\n' + '\n'.join(command_strs)


uas_re = re.compile(r'U: (\d\d.\d+)')
las_re = re.compile(r'L: (\d\d.\d+)')
# TODO: Hook up LAS arg
def _get_acc(text, score='U'):
    if score == 'U':
        return float(uas_re.search(text).groups()[0])
    else:
        return float(las_re.search(text).groups()[0])

