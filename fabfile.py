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


def test1k(model="baseline", dbg=False):
    with lcd(str(LOCAL_REPO)):
        local(_train('~/work_data/stanford/1k_train.txt',  '~/work_data/parsers/tmp',
                    debug=dbg))
        local(_parse('~/work_data/parsers/tmp', '~/work_data/stanford/dev_auto_pos.parse',
                     '/tmp/parse', gold=True))


def draxx_baseline(name):
    model = pjoin(str(REMOTE_PARSERS), name)
    data = str(REMOTE_STANFORD)
    repo = str(REMOTE_REPO)
    train_str = _train(pjoin(data, 'train.txt'), model)
    parse_str = _parse(model, pjoin(data, 'devi.txt'), pjoin(model, 'dev'))
    eval_str = _evaluate(pjoin(model, 'dev', 'parses'), pjoin(data, 'devr.txt'))
    script = _pbsify(repo, [train_str, parse_str, eval_str])
    script_loc = pjoin(repo, 'pbs', '%s_draxx_baseline.pbs' % name)
    with cd(repo):
        put(StringIO(script), script_loc)
        run('qsub -N %s_bl %s' % (name, script_loc))


def draxx_repair(name, extra_feats='False', repairs='True', k=0):
    extra_feats = True if extra_feats == 'True' else False
    repairs = True if repairs == 'True' else False
    k = int(k)
    if extra_feats:
        name += '_x'
    if not repairs:
        name += '_bl'
    if k != 0:
        name += '_k%d' % k
    print name
    data = str(REMOTE_STANFORD)
    repo = str(REMOTE_REPO)
    model_dir = pjoin(str(REMOTE_PARSERS), name)
    repair_str = '-r -d' if repairs else ''
    if repairs:
        train_alg = 'online'
        upd = 'cost'
    elif k == 0:
        train_alg = 'online'
        upd = 'cost'
    else:
        train_alg = 'static'
        upd = 'max'
    try:
        run('mkdir %s' % model_dir)
    except:
        pass
    for i in range(20):
        model = pjoin(model_dir, str(i))
        train_str = _train(pjoin(data, 'train.txt'), model, k=k, i=15,
                           add_feats=extra_feats, train_alg=train_alg,
                           args=repair_str, upd=upd, seed=i)
        parse_str = _parse(model, pjoin(data, 'devi.txt'), pjoin(model, 'dev'), k=k)
        eval_str = _evaluate(pjoin(model, 'dev', 'parses'), pjoin(data, 'devr.txt'))
        script = _pbsify(repo, [train_str, parse_str, eval_str])
        script_loc = pjoin(repo, 'pbs', '%s.pbs' % name)
        with cd(repo):
            put(StringIO(script), script_loc)
            run('qsub -N %s_%d %s' % (name, i, script_loc))

def beam(name, k=8, n=1, size=0, tb='wsj'):
    size = int(size)
    k = int(k)
    n = int(n)
    if tb == 'wsj':
        data = str(REMOTE_STANFORD)
        train_name = 'train.txt'
        eval_pos = 'devi.txt'
        eval_parse = 'devr.txt'

    exp_dir = str(REMOTE_PARSERS)
    train_n(n, name, exp_dir,
            data, k=k, i=15, feat_str="full", train_alg='max', label="Stanford",
            n_sents=size, train_name=train_name, 
            dev_names=(eval_pos, eval_parse))
 



def conll_table(name):
    langs = ['arabic', 'basque', 'catalan', 'chinese', 'czech', 'english',
            'greek', 'hungarian', 'italian', 'turkish']
    systems = ['bl', 'exp']
    for lang in langs:
        bl_accs = []
        exp_accs = []
        for system, accs in zip(systems, ([bl_accs, exp_accs])):

            for i in range(20):
                uas_loc = pjoin(str(REMOTE_PARSERS), 'conll', lang, system,
                                str(i), 'dev', 'acc')
                try:
                    text = run('cat %s' % uas_loc, quiet=True).stdout
                    accs.append(_get_acc(text, score='U'))
                except:
                    continue
        if bl_accs:
            bl_n, bl_acc, stdev = _get_stdev(bl_accs)
        if exp_accs:
            exp_n, exp_acc, stdev = _get_stdev(exp_accs)
        if bl_n == exp_n:
            z, p = scipy.stats.wilcoxon(bl_accs, exp_accs)
        else:
            p = 1.0

        print lang, fmt_pc(bl_acc), fmt_pc(exp_acc), '%.4f' % p

def fmt_pc(pc):
    if pc < 1:
        pc *= 100
    return '%.2f' % pc


def conll(name, lang, n=20, debug=False):
    """Run the 20 seeds for the baseline and experiment conditions for a conll lang"""
    data = str(REMOTE_CONLL)
    repo = str(REMOTE_REPO)
    eval_pos = '%s.test.pos' % lang
    eval_parse = '%s.test.malt' % lang
    train_name = '%s.train.proj.malt' % lang
    n = int(n)
    if debug == True: n = 2
    for condition, arg_str in [('bl', ''), ('exp', '-r -d')]:
        for i in range(n):
            exp_name = '%s_%s_%s_%d' % (name, lang, condition, i)
            model = pjoin(str(REMOTE_PARSERS), name, lang, condition, str(i))
            run("mkdir -p %s" % model)
            train_str = _train(pjoin(data, train_name), model, k=0, i=15,
                               add_feats=False, train_alg='online', seed=i, label="conll",
                               args=arg_str)
            parse_str = _parse(model, pjoin(data, eval_pos), pjoin(model, 'dev'), k=0)
            eval_str = _evaluate(pjoin(model, 'dev', 'parses'), pjoin(data, eval_parse))
            grep_str = "grep 'U:' %s >> %s" % (pjoin(model, 'dev', 'acc'),
                                               pjoin(model, 'dev', 'uas')) 
            script = _pbsify(repo, (train_str, parse_str, eval_str, grep_str))
            if debug:
                print script
                continue
            script_loc = pjoin(repo, 'pbs', exp_name)
            with cd(repo):
                put(StringIO(script), script_loc)
                run('qsub -N %s_bl %s' % (exp_name, script_loc))
 

def ngram_add1(name, k=8, n=1, size=10000):
    import redshift.features
    n = int(n)
    k = int(k)
    size = int(size)
    data = str(REMOTE_MALT)
    repo = str(REMOTE_REPO)
    train_name = '0.train'
    eval_pos = '0.testi' 
    eval_parse = '0.test'
    arg_str = 'full'
    train_n(n, 'base', pjoin(str(REMOTE_PARSERS), name), data, k=k, i=15,
            feat_str="full", train_alg='max', label="NONE", n_sents=size,
            ngrams=0, train_name=train_name)
    tokens = 'S0,N0,N1,N2,N0l,N0l2,S0h,S0h2,S0r,S0r2,S0l,S0l2'.split(',')
    ngram_names = ['%s_%s' % (p) for p in combinations(tokens, 2)]
    ngram_names.extend('%s_%s_%s' % (p) for p in combinations(tokens, 3))
    kernel_tokens = redshift.features.get_kernel_tokens()
    ngrams = list(combinations(kernel_tokens, 2))
    ngrams.extend(combinations(kernel_tokens, 3))
    n_ngrams = len(ngrams)
    n_models = n
    for ngram_id, ngram in list(sorted(enumerate(ngrams))):
        ngram_name = ngram_names[ngram_id]
        train_n(n, '%d_%s' % (ngram_id, ngram_name), pjoin(str(REMOTE_PARSERS), name),
                data, k=k, i=15, feat_str="full", train_alg='max', label="NONE",
                n_sents=size, ngrams='_'.join([str(i) for i in ngram]),
                train_name=train_name, dev_names=(eval_pos, eval_parse))
        n_models += n
        # Sleep 5 mins after submitting 50 jobs
        if n_models > 100:
            time.sleep(350)
            n_models = 0


def combine_ngrams(name, k=8, n=5, size=10000):
    def make_ngram_str(ngrams):
        strings = ['_'.join([str(name_to_idx[t]) for t in ngram.split('_')]) for ngram in ngrams]
        return ','.join(strings)
    n = int(n)
    k = int(k)
    size = int(size)
    data = str(REMOTE_MALT)
    repo = str(REMOTE_REPO)
    train_name = '0.train'
    eval_pos = '0.testi' 
    eval_parse = '0.test'
 
    import redshift.features
    kernel_tokens = redshift.features.get_kernel_tokens()
    token_names = 'S0 N0 N1 N2 N0l N0l2 S0h S0h2 S0r S0r2 S0l S0l2'.split()
    name_to_idx = dict((tok, idx) for idx, tok in enumerate(token_names))
    ngrams = ['S0_S0r_S0l', 'S0_S0h_S0r', 'S0_N1_S0r', 'S0_N0l_S0h', 'S0_N0l_S0r',
            'S0_N0_S0r2', 'S0_N0l2_S0r', 'S0_S0h_S0l', 'S0_N0l2_S0h', 'S0_N0_S0h',
            'S0_S0r_S0l2', 'S0_N0_S0r', 'S0_S0r_S0r2', 'S0_N0_N0l2', 'S0_N1_N0l',
            'S0_N0_N0l', 'S0_S0h_S0r2', 'S0_N1_S0h', 'N0_N0l_S0r2', 'S0_N1_S0r2',
            'N0_N0l_S0r', 'S0_N1_S0l', 'S0_S0h_S0l2', 'S0_N0l_S0r2']
    base_set = []
    n_added = 0
    ngram_str = make_ngram_str(base_set)
    exp_dir = pjoin(str(REMOTE_PARSERS), name, str(n_added))
    n_finished = count_finished(exp_dir)
    if n_finished < n: 
        train_n(n, str(n_added), pjoin(str(REMOTE_PARSERS), name),
                data, k=k, i=15, feat_str="full", train_alg='max', label="NONE",
                n_sents=size, ngrams=0, train_name=train_name, 
                dev_names=(eval_pos, eval_parse))
        n_finished = 0
        while n_finished < n:
            time.sleep(60)
            n_finished = count_finished(exp_dir)
    base_accs = get_accs(exp_dir)
    base_avg = sum(base_accs) / len(base_accs)
    print "Base: ", base_avg
    rejected = []
    while True:
        next_ngram = ngrams.pop(0)
        n_added += 1
        print "Testing", next_ngram
        ngram_str = make_ngram_str(base_set + [next_ngram])
        exp_dir = pjoin(str(REMOTE_PARSERS), name, str(n_added))
        n_finished = count_finished(exp_dir)
        if n_finished < n:
            train_n(n, str(n_added), pjoin(str(REMOTE_PARSERS), name),
                    data, k=k, i=15, feat_str="full", train_alg='max',
                    label="NONE", n_sents=size, ngrams=ngram_str, train_name=train_name,
                    dev_names=(eval_pos, eval_parse))
            n_finished = 0
            while n_finished < n:
                time.sleep(60)
                n_finished = count_finished(exp_dir)
        exp_accs = get_accs(exp_dir)
        exp_avg = sum(exp_accs) / len(exp_accs)
        if n >= 20:
            _, p = scipy.stats.wilcoxon(exp_accs, base_accs)
        else:
            p = 0.0
        if exp_avg > base_avg and p < 0.1:
            print "Accepted!", next_ngram, base_avg, exp_avg, p
            base_set.append(next_ngram)
            base_avg = exp_avg
            base_accs = exp_accs
        else:
            print "Rejected!", next_ngram, base_avg, exp_avg, p
            rejected.append(next_ngram)
        print "Current set: ", ' '.join(base_set)
        print "Rejected:", ' '.join(rejected)

def get_best_trigrams(all_trigrams, n=25):
    best = [2, 199, 158, 61, 66, 5, 150, 1, 88, 154, 85, 25, 53, 10, 3, 60, 73,
            175, 114, 4, 6, 148, 205, 197, 0, 71, 127, 200, 142, 84, 43, 89, 45,
            95, 419, 33, 110, 182, 20, 24, 159, 51, 106, 26, 8, 178, 151, 12, 166,
            192, 7, 209, 190, 147, 13, 194, 50, 129, 174, 186, 28, 116, 193, 179,
            262, 23, 44, 172, 133, 191, 562, 38, 124, 195, 123, 72, 202, 187, 101,
            92, 104, 115, 596, 29, 99, 132, 169, 42, 206, 592, 67, 323, 69, 9, 74,
            14, 136, 64, 561, 161, 19, 77, 171, 300, 204, 310, 121, 15, 201, 235,
            657, 70, 198, 22, 68, 48, 153, 54, 286, 83, 162, 100, 506, 98, 80, 433,
            420, 63, 613, 149, 90, 139, 31, 91, 86, 203, 248, 173, 130, 165, 346,
            157, 616, 18, 145, 451, 410, 75, 55, 603, 156, 52, 622, 210, 332, 120]
 

def tritable(name):
    #exp_dir = REMOTE_PARSERS.join(name)
    exp_dir = Path('/data1/mhonniba/').join(name)
    results = []
    with cd(str(exp_dir)):
        ngrams = run("ls %s" % exp_dir, quiet=True).split()
        for ngram in sorted(ngrams):
            base_dir = exp_dir.join(ngram).join('base')
            tri_dir = exp_dir.join(ngram).join('exp')
            base_accs = get_accs(str(base_dir))
            tri_accs = get_accs(str(tri_dir))
            if not base_accs or not tri_accs:
                continue
            if len(base_accs) != len(tri_accs):
                continue
            #z, p = scipy.stats.wilcoxon(base_accs, tri_accs)
            p = 1.0
            delta =  (sum(tri_accs) / len(tri_accs)) - (sum(base_accs) / len(base_accs))
            results.append((delta, ngram, p))
        results.sort(reverse=True)
        good_trigrams = []
        for delta, ngram, p in results:
            ngram = ngram.replace('s0le', 'n0le')
            pieces = ngram.split('_')
            print r'%s & %s & %s & %.1f \\' % (pieces[1], pieces[2], pieces[3], delta)
            if delta > 0.1:
                good_trigrams.append(int(ngram.split('_')[0]))
        print good_trigrams
        print len(good_trigrams)
            

def bitable(name):
    exp_dir = REMOTE_PARSERS.join(name)
    base_accs = get_accs(str(exp_dir.join('0_S0_N0')))
    base_acc = sum(base_accs) / len(base_accs)
    print "Base:", len(base_accs), sum(base_accs) / len(base_accs)
    results = []
    with cd(str(exp_dir)):
        ngrams = run("ls %s" % exp_dir, quiet=True).split()
        for ngram in sorted(ngrams):
            if ngram == 'base' or ngram == '0_S0_N0':
                continue
            accs = get_accs(str(exp_dir.join(ngram)))
            print ngram, len(accs)
            if not accs:
                continue
            _, avg, stdev = _get_stdev(accs)
            z, p = scipy.stats.wilcoxon(accs, base_accs)
            parts = ngram.split('_')
            if ngram.startswith('base'):
                base_acc = avg
            else:
                results.append((avg, ngram, stdev, p))
    good_ngrams = []
    results.sort()
    results.reverse()
    for acc, ngram, stdev, p in results:
        ngram = '_'.join(ngram.split('_')[1:])
        if acc > base_acc and p < 0.01:
            print r'%s & %.3f & %.3f \\' % (ngram, acc - base_acc, p)
            good_ngrams.append(ngram)
    print good_ngrams
    print len(good_ngrams)
        

def vocab_thresholds(name, k=8, n=1, size=10000):
    base_dir = REMOTE_PARSERS.join(name)
    n = int(n)
    k = int(k)
    size = int(size)
    data = str(REMOTE_STANFORD)
    repo = str(REMOTE_REPO)
    train_name = 'train.txt'
    eval_pos = 'devi.txt' 
    eval_parse = 'devr.txt'
 
    thresholds = [75]
    ngram_sizes = [60, 90, 120]
    for n_ngrams in ngram_sizes:
        if n_ngrams == 0:
            feat_name = 'zhang'
        else:
            feat_name = 'full'
        exp_dir = str(base_dir.join('%d_ngrams' % n_ngrams))
        #if n_ngrams < 100:
        #    train_n(n, 'unpruned', exp_dir, data, k=k, i=15, t=0, f=0,
        #            train_alg="max", label="Stanford", n_sents=size, feat_str=feat_name)
        for t in thresholds:
            thresh = 'thresh%d' % t
            train_n(n, thresh, exp_dir, data, k=k, i=15, t=t, f=100,
                    train_alg='max', label="Stanford", n_sents=size,
                    feat_str=feat_name, ngrams=n_ngrams)

def vocab_table(name):
    exp_dir = REMOTE_PARSERS.join(name)
    with cd(str(exp_dir)):
        conditions = run("ls %s" % exp_dir, quiet=True).split()
        for condition in sorted(conditions):
            accs = get_accs(str(exp_dir.join(condition)))
            print condition, len(accs), sum(accs) / len(accs)

# 119_s0_s0r2_s0l2
def train_n(n, name, exp_dir, data, k=1, feat_str="zhang", i=15, upd='max',
            train_alg="online", label="Stanford", n_sents=0,
            ngrams=0, t=0, f=0, train_name='train.txt', dev_names=('devi.txt', 'devr.txt')):
    exp_dir = str(exp_dir)
    repo = str(REMOTE_REPO)
    for seed in range(n):
        exp_name = '%s_%d' % (name, seed)
        model = pjoin(exp_dir, name, str(seed))
        run("mkdir -p %s" % model, quiet=True)
        train_str = _train(pjoin(data, train_name), model, k=k, i=15,
                           feat_str=feat_str, train_alg=train_alg, seed=seed,
                           label=label, n_sents=n_sents, ngrams=ngrams,
                           vocab_thresh=t, feat_thresh=f)
        parse_str = _parse(model, pjoin(data, dev_names[0]), pjoin(model, 'dev'))
        eval_str = _evaluate(pjoin(model, 'dev', 'parses'), pjoin(data, dev_names[1]))
        grep_str = "grep 'U:' %s >> %s" % (pjoin(model, 'dev', 'acc'),
                                           pjoin(model, 'dev', 'uas')) 
        # Save disk space by removing models
        del_str = "rm %s %s" % (pjoin(model, "model"), pjoin(model, "words"))
        script = _pbsify(repo, (train_str, parse_str, eval_str, grep_str, del_str))
        script_loc = pjoin(repo, 'pbs', exp_name)
        with cd(repo):
            put(StringIO(script), script_loc)
            err_loc = pjoin(model, 'stderr')
            out_loc = pjoin(model, 'stdout')
            run('qsub -N %s %s -e %s -o %s' % (exp_name, script_loc, err_loc, out_loc), quiet=True)


def count_finished(exp_dir):
    with cd(exp_dir):
        samples = [s for s in run("ls %s/*/" % exp_dir, quiet=True).split()
                   if s.endswith('stdout')]
    return len(samples)


def get_accs(exp_dir, eval_name='dev'):
    results = []
    with cd(exp_dir):
        results = [float(s.split()[1]) for s in
                   run("grep 'U:' %s/*/dev/acc" % exp_dir, quiet=True).split('\n')
                   if s.strip()]
    return results


def _train(data, model, debug=False, k=1, feat_str='zhang', i=15, upd='early',
           train_alg="online", seed=0, args='', label="Stanford",
           n_sents=0, ngrams=0, vocab_thresh=0, feat_thresh=10):
    template = './scripts/train.py -i {i} -a {alg} -k {k} -x {feat_str} {data} {model} -s {seed} -l {label} -n {n_sents} -g {ngrams} -t {vocab_thresh} -f {feat_thresh} {args}'
    if debug:
        template += ' -debug'
    return template.format(data=data, model=model, k=k, feat_str=feat_str, i=i,
                           vocab_thresh=vocab_thresh, feat_thresh=feat_thresh,
                          upd=upd, alg=train_alg, seed=seed,
                          label=label, args=args, n_sents=n_sents, ngrams=ngrams)


def _parse(model, data, out, gold=False):
    template = './scripts/parse.py {model} {data} {out} '
    if gold:
        template += '-g'
    return template.format(model=model, data=data, out=out)


def _evaluate(test, gold):
    return './scripts/evaluate.py %s %s > %s' % (test, gold, test.replace('parses', 'acc'))


def _pbsify(repo, command_strs):
    header = """#! /bin/bash
#PBS -l walltime=20:00:00,mem=2gb,nodes=1:ppn=2
source /home/mhonniba/ev/bin/activate
export PYTHONPATH={repo}:{repo}/redshift:{repo}/svm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/lib64:/usr/lib64/:/usr/lib64/atlas:{repo}/redshift/svm/lib/
cd {repo}"""
    return header.format(repo=repo) + '\n' + '\n'.join(command_strs)


uas_re = re.compile(r'U: (\d\d.\d+)')
las_re = re.compile(r'L: (\d\d.\d+)')
# TODO: Hook up LAS arg
def _get_acc(text, score='U'):
    if score == 'U':
        return float(uas_re.search(text).groups()[0])
    else:
        return float(las_re.search(text).groups()[0])


def _get_stdev(scores):
    n = len(scores)
    mean = sum(scores) / n
    var = sum((s - mean)**2 for s in scores)/n
    return n, mean, sqrt(var)

def _get_repair_str(reattach, lower, invert):
    repair_str = []
    if reattach:
        repair_str.append('-r -o')
    if lower:
        repair_str.append('-w')
    if invert:
        repair_str.append('-v')
    return ' '.join(repair_str)


def _get_paths(here):
    if here == True:
        return LOCAL_REPO, LOCAL_STANFORD, LOCAL_PARSERS
    else:
        return REMOTE_REPO, REMOTE_STANFORD, REMOTE_PARSERS


def _get_train_name(data_loc, size):
    if size == 'full':
        train_name = 'train.txt'
    elif size == '1k':
        train_name = '1k_train.txt'
    elif size == '5k':
        train_name = '5k_train.txt'
    elif size == '10k':
        train_name = '10k_train.txt'
    else:
        raise StandardError(size)
    return data_loc.join(train_name)


def run_static(name, size='full', here=True, feats='all', labels="MALT", thresh=5, reattach=False,
              lower=False):
    train_name = _get_train_name(size)
    repair_str = ''
    if reattach:
        repair_str += '-r '
    if lower:   
        repair_str += '-m'
    if feats == 'all':
        feats_flag = ''
    elif feats == 'zhang':
        feats_flag = '-x'
    if here is True:
        data_loc = Path(LOCAL_STANFORD)
        #if labels == 'Stanford':
        #    data_loc = Path(LOCAL_STANFORD)
        #else:
        #    data_loc = Path(LOCAL_CONLL)
        parser_loc = Path(LOCAL_PARSERS).join(name)
        runner = local
        cder = lcd
        repo = LOCAL_REPO
    else:
        if labels == 'Stanford':
            data_loc = Path(REMOTE_STANFORD)
        else:
            data_loc = Path(REMOTE_CONLL)
        parser_loc = Path(REMOTE_PARSERS).join(name)
        runner = run
        cder = cd
        repo = REMOTE_REPO

    train_loc = data_loc.join(train_name)
    with cder(repo):
        #runner('make -C redshift clean')
        runner('make -C redshift')
        if here is not True:
            arg_str = 'PARSER_DIR=%s,DATA_DIR=%s,FEATS="%s,LABELS=%s,THRESH=%s,REPAIRS=%s"' % (parser_loc, data_loc, feats_flag, labels, thresh, repair_str)
            job_name = 'redshift_%s' % name
            err_loc = parser_loc.join('err')
            out_loc = parser_loc.join('log')
            run('qsub -e %s -o %s -v %s -N %s pbs/redshift.pbs' % (err_loc, out_loc, arg_str, job_name))
            print "Waiting 2m for job to initialise"
            time.sleep(120)
            run('qstat -na | grep mhonniba')
            if err_loc.exists():
                print err_loc.open()

        else:
            dev_loc = data_loc.join('devr.txt')
            in_loc = data_loc.join('dev_auto_pos.parse')
            out_dir = parser_loc.join('parsed_dev')
            runner('./scripts/train.py %s -f %d -l %s %s %s %s' % (repair_str, thresh, labels, feats_flag, train_loc, parser_loc))
            runner('./scripts/parse.py -g %s %s %s' % (parser_loc, in_loc, out_dir))
            runner('./scripts/evaluate.py %s %s' % (out_dir.join('parses'), dev_loc)) 
