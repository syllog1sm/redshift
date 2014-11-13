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

c_options = {
    'transition_system': 'arc_eager'
}
with open(path.join(pwd, 'redshift', 'compile_time_options.pxi'), 'w') as file_:
    for k, v in c_options.iteritems():
        file_.write("DEF %s = '%s'\n" % (k.upper(), v))

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
   Extension("index.hashes", ["index/hashes.pyx"], language="c++",
              include_dirs=includes),
    Extension("index.lexicon", ["index/lexicon.pyx"], language="c++",
              include_dirs=includes),
    Extension("redshift.tagger", ["redshift/tagger.pyx"], include_dirs=includes,
              language="c++"),
    Extension("redshift._tagger_features", ["redshift/_tagger_features.pyx"],
              include_dirs=includes, language="c++"),
    Extension("redshift.pystate", ["redshift/pystate.pyx"], include_dirs=includes,
              language="c++"),
]

# The parser chooses one of these by a value in compile_time_options.pxi
if c_options['transition_system'] == 'arc_eager':
    exts.append(
        Extension('redshift.arc_eager', ["redshift/arc_eager.pyx"],
            language="c++", include_dirs=includes),
    )
elif c_options['transition_system'] == 'arc_hybrid':
    exts.append(
        Extension('redshift.arc_hybrid', ["redshift/arc_hybrid.pyx"],
            language="c++", include_dirs=includes),
    )
else:
    raise StandardError('Unknown transition system: %s' % c_options['transition_system'])


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
    license='Trial. Contact for full commercial license',
    name='redshift-parser',
    packages=['redshift', 'index'],
    package_data={'redshift': ['*.pxd'], 'index': ['*.pxd', 'english.case', 'bllip-clusters']},
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    version='1.0',
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=exts,
    url='https://github.com/syllog1sm/redshift',
)



