from fabric.api import local, run, lcd, cd, env
from fabric.operations import get, put
from fabric.contrib.files import exists
import time
import re
from math import sqrt
from os import path
from os import listdir


try:
    from _secret_paths import *
except ImportError:
    print "Tip: Make a _secret_paths.py file to set up paths."
    print "_secret_paths.py is in the .gitignore, so won't be commited."
    HOSTS = []
    GATEWAY = ''
    LOCAL_REPO = path.dirname(__file__)
    REMOTE_REPO = None

try:
    from fabfiles.exp import *
except:
    pass

env.use_ssh_config = True


def clean():
    local('python setup.py clean --all')

def make():
    local('python setup.py build_ext --inplace > /tmp/stdout 2> /tmp/stderr')

def vmake():
    local('python setup.py build_ext --inplace')

def test():
    local('py.test')

def qstat():
    run("qstat -na | grep mhonn")


def deploy():
    clean()
    make()
    with cd(str(REMOTE_REPO)):
        run('git pull')
