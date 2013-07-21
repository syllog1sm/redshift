import Cython.Distutils
from distutils.extension import Extension
import distutils.core

import sys
import os
from os.path import join as pjoin
import numpy


def clean(ext):
    for pyx in ext.sources:
        if pyx.endswith('.pyx'):
            c = pyx[:-4] + '.c'
            cpp = pyx[:-4] + '.cpp'
            so = pyx[:-4] + '.so'
            if os.path.exists(so):
                os.unlink(so)
            if os.path.exists(c):
                os.unlink(c)
            elif os.path.exists(cpp):
                os.unlink(cpp)


pwd = os.path.dirname(__file__)
virtual_env = os.environ.get('VIRTUAL_ENV', '')

includes = [numpy.get_include(),
            os.path.join(virtual_env, 'include'),
            os.path.join(pwd, 'ext')]
libs = [os.path.join(pwd, 'ext')]

exts = [
    Extension('ext.murmurhash', ["ext/murmurhash.pyx", "ext/MurmurHash2.cpp",
              "ext/MurmurHash3.cpp"], language="c++", include_dirs=includes),
    Extension('ext.sparsehash', ["ext/sparsehash.pyx"], language="c++",
              include_dirs=includes),
    Extension('redshift.parser', ["redshift/parser.pyx"], language="c++",
              include_dirs=includes),
    Extension('redshift.beam', ["redshift/beam.pyx"], language="c++",
              include_dirs=includes),
    Extension('redshift._state', ["redshift/_state.pyx"], language="c++", include_dirs=includes),
    Extension('redshift.io_parse', ["redshift/io_parse.pyx"], language="c++",
               include_dirs=includes),
    Extension('redshift.features', ["redshift/features.pyx"],
        language="c++", include_dirs=includes),
    Extension('redshift.transitions', ["redshift/transitions.pyx"],
        language="c++", include_dirs=includes),
    Extension('learn.perceptron', ['learn/perceptron.pyx'], language="c++",
              include_dirs=includes),
    Extension("index.hashes", ["index/hashes.pyx", "ext/MurmurHash2.cpp",
                               "ext/MurmurHash3.cpp"], language="c++",
              include_dirs=includes)
]

if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)

distutils.core.setup(
    name='Redshift',
    packages=['redshift'],
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    version='1.0',
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts
)



