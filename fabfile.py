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

env.use_ssh_config = True

from _paths import REMOTE_REPO, REMOTE_MALT, REMOTE_STANFORD, REMOTE_PARSERS
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
        err = local('python setup.py build_ext --inplace', capture=True).stderr
        for line in err:
            if 'error: ' in line:
                print line

def deploy():
    clean()
    make()
    with cd(REMOTE_REPO):
        run('git pull')

def draxx_baseline(name):
    model = pjoin(str(REMOTE_PARSERS), name)
    data = str(REMOTE_STANFORD)
    repo = str(REMOTE_REPO)
    train_str = _train(pjoin(data, 'train.txt'), model)
    parse_str = _parse(model, pjoin(data, 'devi.txt'), pjoin(model, 'dev'))
    eval_str = _evaluate(pjoin(model, 'dev', 'parses'), pjoin(data, 'devr.text'))
    script = _pbsify(repo, [train_str, parse_str, eval_str])
    script_loc = pjoin(repo, 'pbs', '%s_draxx_baseline.pbs' % name)
    with cd(repo):
        put(StringIO(script), script_loc)
        #run('qsub -N %s_bl %s' % (name, script_loc))

def test1k(model="baseline", dbg=False):
    with lcd(str(LOCAL_REPO)):
        local(_train('~/work_data/stanford/1k_train.txt',  '~/work_data/parsers/tmp',
                    debug=dbg))
        local(_parse('~/work_data/parsers/tmp', '~/work_data/stanford/dev_auto_pos.parse',
                     '/tmp/parse', gold=True))



def _pbsify(repo, command_strs):
    header = """#! /bin/bash
#PBS -l walltime=20:00:00,mem=8gb,nodes=1:ppn=5
source /home/mhonniba/py27/bin/activate
export PYTHONPATH={repo}:{repo}/redshift:{repo}/svm
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:/usr/lib64/:/usr/local/lib64/:/usr/lib64/atlas:{repo}/redshift/svm/lib/
cd {repo}"""
    return header.format(repo=repo) + '\n' + '\n'.join(command_strs)

def _train(data, model, debug=False):
    template = './scripts/train.py {data} {model}'
    if debug:
        template += ' -debug'
    return template.format(data=data, model=model)

def _parse(model, data, out, gold=False):
    template = './scripts/parse.py {model} {data} {out} '
    if gold:
        template += '-g'
    return template.format(model=model, data=data, out=out)

def _evaluate(test, gold):
    return './scripts/evaluate.py %s %s' % (test, gold)

def avg_accs(exp_name, test=False):
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
            samples = run("ls %s" % sys_dir).split()
            for sample in samples:
                if sample == 'logs':
                    continue
                sample = Path(sys_dir).join(sample)
                if test:
                    acc_loc = sample.join('test').join('acc')
                else:
                    acc_loc = sample.join('dev').join('acc')
                try:
                    text = run("cat %s" % acc_loc).stdout
                except:
                    continue
                uas.append(_get_acc(text, score='U'))
                las.append(_get_acc(text, score='L'))
            uas_results[system] = _get_stdev(uas)
            las_results[system] = _get_stdev(las)
    return uas_results, las_results

def dev_eval_online(exp_name):
    uas_results, las_results = avg_accs(exp_name)
    print "UAS"
    for name, (n, mean, stdev) in uas_results.items():
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    print "LAS"
    for name, (n, mean, stdev) in las_results.items():
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)


def eval_online(exp_name):
    uas_results, las_results = avg_accs(exp_name, test=True)
    print "UAS"
    for name, (n, mean, stdev) in uas_results.items():
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)
    print "LAS"
    for name, (n, mean, stdev) in las_results.items():
        print '%s %d: %.1f +/- %.2f' % (name, n, mean, stdev)




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


