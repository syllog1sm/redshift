from fabric.api import local, run, lcd, cd, env
from pathlib import Path
import time

env.use_ssh_config = True

from _paths import REMOTE_REPO, REMOTE_MALT, REMOTE_STANFORD, REMOTE_PARSERS
from _paths import LOCAL_REPO, LOCAL_MALT, LOCAL_STANFORD, LOCAL_PARSERS
from _paths import HOSTS, GATEWAY

env.hosts = HOSTS
env.gateway = GATEWAY

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


def run_static(name, size='full', here=None, feats='all', labels="MALT", thresh=5, reattach=False,
              lower=False):
    if here == 'True':
        here = True
    elif here == 'False':
        here = False
    if size == 'full':
        train_name = 'train.txt'
        if here is None:
            here = False
    elif size == '1k':
        train_name = '1k_train.txt'
        if here is None:
            here = True
    elif size == '5k':
        train_name = '5k_train.txt'
        if here is None:
            here = True
    elif size == '10k':
        train_name = '10k_train.txt'
        if here is None:
            here = False
    else:
        raise StandardError(size)
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
        if labels == 'Stanford':
            data_loc = Path(LOCAL_STANFORD)
        else:
            data_loc = Path(LOCAL_CONLL)
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
        runner('make -C redshift clean')
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
            runner('./redshift/train.py -f %d -l %s %s %s %s' % (thresh, labels, feats_flag, train_loc, parser_loc))
            runner('./redshift/parse.py -g %s %s %s' % (parser_loc, in_loc, out_dir))
            runner('./redshift/evaluate.py %s %s' % (out_dir.join('parses'), dev_loc)) 
