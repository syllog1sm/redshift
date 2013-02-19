from fabric.api import local, run, lcd, cd, env
from fabric.operations import get
from pathlib import Path
import time
import re
from math import sqrt
from os.path import join as pjoin
from os import listdir
import scipy.stats

env.use_ssh_config = True

from _paths import REMOTE_REPO, REMOTE_MALT, REMOTE_STANFORD, REMOTE_PARSERS
from _paths import LOCAL_REPO, LOCAL_MALT, LOCAL_STANFORD, LOCAL_PARSERS
from _paths import HOSTS, GATEWAY

env.hosts = HOSTS
env.gateway = GATEWAY


def recompile(runner=local):
    runner("make -C redshift clean")
    runner("make -C index clean")
    runner("make -C svm clean")
    runner("make -C svm")
    runner("make -C index")
    runner("make -C redshift")


def deploy():
    local("make -C redshift")
    local("git push")
    with cd(REMOTE_REPO):
        run('git pull')


def amend(target="."):
    local("git add %s" % target)
    local('git commit -m "* Amendment"')
    local("git pull")
    deploy()

def reattach(name, size="5k", feats='base', thresh=5, here=True, n=1, niters=15):
    exp(name, size=size, feats=feats, thresh=thresh, here=here, n=n,
        reattach=True, niters=niters)


def invert(name, size="5k", feats='base', thresh=5, here=True, n=1, niters=15):
    exp(name, size=size, feats=feats, thresh=thresh, here=here, n=n,
        invert=True, reattach=True, niters=niters)


def lower(name, size="5k", feats='base', thresh=5, here=True, n=1, niters=15):
    exp(name, size=size, feats=feats, thresh=thresh, here=here, n=n,
        lower=True, reattach=True, niters=niters)

def full(name, size="5k", feats='base', thresh=5, here=True, n=1, niters=15):
    exp(name, size=size, feats=feats, thresh=thresh, here=here, n=n,
        lower=True, invert=True, reattach=True, niters=niters)


def exp(name, size="5k", feats='base', thresh=5, reattach=False, lower=False,
        invert=False, extra=False, here=True, n=1, niters=15):
    runner = local if here == True else remote
    cder = lcd if here == True else cd
    recompile(runner)
    repair_str = _get_repair_str(reattach, lower, invert)
    feat_flag = '-x' if feats == 'extra' else ''
    repo, data_loc, parser_loc = _get_paths(here)
    train_loc= _get_train_name(data_loc, size)
    parser_loc = parser_loc.join(name)
    if not parser_loc.exists():
        parser_loc.mkdir()
    dev_loc = data_loc.join('devr.txt')
    in_loc = data_loc.join('dev_auto_pos.parse')
    train_str = './scripts/train.py -s {seed} {repair} -i {niters} -f {thresh} {feats} {train} {out}'
    parse_str = './scripts/parse.py -g {parser} {text} {parses}'
    eval_str = './scripts/evaluate.py {parse_loc} {dev_loc} > {out_loc}'
    with cder(str(repo)):
        accs = []
        for i in range(int(n)):
            if n > 1:
                model_dir = parser_loc.join(str(i))
            else:
                model_dir = parser_loc
            out_dir = model_dir.join('dev')
            runner(train_str.format(seed=i, repair=repair_str, niters=niters, thresh=thresh,
                                    feats=feat_flag, train=train_loc, out=model_dir))
            runner(parse_str.format(parser=model_dir, text=in_loc, parses=out_dir))
            runner(eval_str.format(parse_loc=out_dir.join('parses'), dev_loc=dev_loc,
                                   out_loc=out_dir.join('acc')))
            accs.append(_get_acc(out_dir.join('acc')))
    print ', '.join('%.2f' % a for a in accs)
    mean = sum(accs)/len(accs)
    var = sum((a - mean)**2 for a in accs)/len(accs)
    print '%.2f +/- %.2f' % (mean, sqrt(var))


def get_accs(exp_name, test=False):
    exp_dir = pjoin(str(REMOTE_PARSERS), exp_name)
    if test:
        exps = ['baseline', 'both']
    else:
        exps = ['baseline', 'reattach', 'adduce', 'both']
    uas_results = {}
    las_results = {}
    with cd(exp_dir):
        for system in exps:
            sys_dir = pjoin(exp_dir, system)
            uas = []
            las = []
            usent = []
            lsent = []
            samples = sorted(run("ls %s" % sys_dir, quiet=True).split())
            for sample in samples:
                if sample == 'logs':
                    continue
                sample = Path(sys_dir).join(sample)
                print sample
                if test:
                    acc_loc = sample.join('test').join('acc')
                else:
                    acc_loc = sample.join('dev').join('acc')
                text = run("cat %s" % acc_loc, quiet=True).stdout
                uas.append(_get_acc(text, score='U'))
                las.append(_get_acc(text, score='L'))
                usent.append(_get_acc(text, score='u'))
                lsent.append(_get_acc(text, score='l'))

            uas_results[system] = (uas, usent)
            las_results[system] = (las, lsent)
    return uas_results, las_results


def dev_eval_online(exp_name):
    uas_results, las_results = get_accs(exp_name)
    print "UAS"
    for name, (accs, _) in uas_results.items():
        n, mean, stdev = _get_stdev(accs)
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    print "LAS"
    for name, (accs, _) in las_results.items():
        n = len(accs)
        n, mean, stdev = _get_stdev(accs)
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    print "U Sent %"
    for name, (_, accs) in uas_results.items():
        n, mean, stdev = _get_stdev(accs)
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    print "L Sent %"
    for name, (_, accs) in las_results.items():
        n, mean, stdev = _get_stdev(accs)
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)

def textab(stan_name, malt_name):
    def fmt_acc(accs):
        if accs == 0:
            return '00.0'
        n, mean, stdev = _get_stdev(accs)
        return '%.1f' % mean

    zeroes = {'baseline': (0, 0), 'reattach': (0, 0), 'adduce': (0, 0), 'both': (0, 0)}
    if stan_name == 'None':
        stan_u = zeroes
        stan_l = zeroes
    else:
        stan_u, stan_l = get_accs(stan_name)
    if malt_name == 'None':
        malt_u = zeroes
        malt_l = zeroes
    else:
        malt_u, malt_l = get_accs(malt_name)
    exps = [('Baseline', 'baseline'), ('NM L', 'reattach'), ('NM D', 'adduce'),
            ('NM L+D', 'both')]
    print r"""\hline \hline
        & \multicolumn{4}{c}{Unlabelled Attachment} \\
        \hline"""
    for stan, malt in [(stan_u, malt_u), (stan_l, malt_l)]:
        for paper_name, exp_name in exps:
            stan_w, stan_s = stan[exp_name]
            malt_w, malt_s = malt[exp_name]
            row = [paper_name]
            row.extend(map(fmt_acc, (stan_w, stan_s, malt_w, malt_s)))
            print ' & '.join(row), r'\\'
        print r"""\hline
            & \multicolumn{4}{c}{Labelled Attachment} \\
            \hline"""




def eval_online(exp_name):
    uas_results, las_results = get_accs(exp_name, test=True)
    print "UAS"
    for name, (accs, _) in uas_results.items():
        n, mean, stdev = _get_stdev(accs)
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    acc1, acc2 = uas_results.values()
    print "LAS"
    for name, (accs, _) in las_results.items():
        n, mean, stdev = _get_stdev(accs)
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    acc1, acc2 = las_results.values()


def _get_acc(text, score='U'):
    score_re = re.compile(r'%s: (\d\d.\d\d\d)' % score)
    return float(score_re.search(text).groups()[0])


def _get_stdev(scores):
    scores = [s for s in scores if s is not None]
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


