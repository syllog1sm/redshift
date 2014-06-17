#!/usr/bin/env python
import Cython.Distutils
from distutils.extension import Extension
import distutils.core

import sys
import os
from os.path import join as pjoin


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

includes = [os.path.join(virtual_env, 'include'),
            os.path.join(pwd, 'include'),
            os.path.join(pwd, 'ext'),
            os.path.join(pwd, 'ext/include')]
libs = [os.path.join(pwd, 'ext')]

compile_args = ['-fopenmp']
link_args = ['-fopenmp']

exts = [
    Extension('ext.murmurhash', ["ext/murmurhash.pyx", "ext/MurmurHash2.cpp",
              "ext/MurmurHash3.cpp"], language="c++", include_dirs=includes,
              extra_compile_args=compile_args, extra_link_args=link_args),
    Extension('ext.sparsehash', ["ext/sparsehash.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift.parser', ["redshift/parser.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift.nbest_parser', ["redshift/nbest_parser.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
 
    Extension('redshift.beam', ["redshift/beam.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift._state', ["redshift/_state.pyx", "ext/MurmurHash2.cpp",
                                  "ext/MurmurHash3.cpp"],
              language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift.sentence', ["redshift/sentence.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift.lattice_utils', ["redshift/lattice_utils.pyx"], language="c++",
               include_dirs=includes,
               extra_compile_args=compile_args,
               extra_link_args=link_args),
    Extension('redshift.sentence', ["redshift/sentence.pyx"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift._parse_features', ["redshift/_parse_features.pyx"],
              language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension('redshift._lattice_features', ["redshift/_lattice_features.pyx"],
              language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
 
    Extension('redshift.transitions', ["redshift/transitions.pyx"],
        language="c++", include_dirs=includes,
        extra_compile_args=compile_args,
        extra_link_args=link_args),
    Extension('learn.perceptron', ['learn/perceptron.pyx'], language="c++",
              include_dirs=includes,
              extra_compile_args=['-O3'] + compile_args,
              extra_link_args=['-O3'] + link_args),
    Extension("index.hashes", ["index/hashes.pyx", "ext/MurmurHash2.cpp",
                               "ext/MurmurHash3.cpp"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension("index.lexicon", ["index/lexicon.pyx", "ext/MurmurHash2.cpp",
                               "ext/MurmurHash3.cpp"], language="c++",
              include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),

    Extension("features.extractor", ["features/extractor.pyx", "ext/MurmurHash2.cpp",
              "ext/MurmurHash3.cpp"], language="c++", include_dirs=includes,
              extra_compile_args=compile_args,
              extra_link_args=link_args),
    Extension("redshift.tagger", ["redshift/tagger.pyx", "ext/MurmurHash2.cpp",
                                  "ext/MurmurHash3.cpp"], include_dirs=includes,
        language="c++", extra_compile_args=compile_args, extra_link_args=link_args),
    #Extension("redshift.tester", ["redshift/tester.pyx"], include_dirs=includes)
]


if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)

distutils.core.setup(
    name='Redshift shift-reduce dependency parser',
    packages=['redshift'],
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    version='1.0',
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts,
)



