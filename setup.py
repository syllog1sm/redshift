#!/usr/bin/env python
import Cython.Distutils
from distutils.extension import Extension
import distutils.core

import sys
import os
from os.path import join as pjoin
from os import path
from glob import glob


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

includes = []

if 'VIRTUAL_ENV' in os.environ:
    includes += glob(path.join(os.environ['VIRTUAL_ENV'], 'include', 'site', '*'))
else:
    # If you're not using virtualenv, set your include dir here.
    pass

libs = []

exts = [
    Extension('redshift.parser', ["redshift/parser.pyx"], language="c++",
              include_dirs=includes),
    Extension('redshift.beam', ["redshift/beam.pyx"], language="c++",
              include_dirs=includes),
    Extension('redshift._state', ["redshift/_state.pyx"],
                                  language="c++", include_dirs=includes),
    Extension('redshift.sentence', ["redshift/sentence.pyx"], language="c++",
               include_dirs=includes),

    Extension('redshift.sentence', ["redshift/sentence.pyx"], language="c++",
              include_dirs=includes),
    Extension('redshift._parse_features', ["redshift/_parse_features.pyx"],
              language="c++", include_dirs=includes),
    Extension('redshift.transitions', ["redshift/transitions.pyx"],
        language="c++", include_dirs=includes),
    Extension("index.hashes", ["index/hashes.pyx"], language="c++",
              include_dirs=includes),
    Extension("index.lexicon", ["index/lexicon.pyx"], language="c++",
              include_dirs=includes),

    Extension("features.extractor", ["features/extractor.pyx"],
              language="c++", include_dirs=includes),
    Extension("redshift.tagger", ["redshift/tagger.pyx"], include_dirs=includes,
              language="c++"),
    Extension("learn.thinc", ["learn/thinc.pyx"], include_dirs=includes, language="c++",
              extra_compile_args=['-O2'], extra_link_args=['-O2']), 
 
    #Extension("redshift.tester", ["redshift/tester.pyx"], include_dirs=includes)
]


if sys.argv[1] == 'clean':
    print >> sys.stderr, "cleaning .c, .c++ and .so files matching sources"
    map(clean, exts)

distutils.core.setup(
    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],
    description='Redshift shift-reduce dependency parser',
    keywords='natural-language syntactic dependency parser',
    long_description=open('README.rst').read(),
    license='GPL / Contact for commercial',
    name='redshift-parser',
    packages=['redshift'],
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    version='1.0',
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts,
    url='https://github.com/syllog1sm/redshift',
)



