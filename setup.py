import Cython.Distutils
from distutils.extension import Extension
import distutils.core

import os
from os.path import join as pjoin
import numpy


LIBLINEAR = 'liblinear-weights-1.91'
pwd = os.path.dirname(__file__)

virtual_env = os.environ.get('VIRTUAL_ENV', '')

distutils.core.setup(
    name='Redshift',
    packages=['redshift'],
    author='Matthew Honnibal',
    author_email='honnibal@gmail.com',
    version='1.0',
    cmdclass={'build_ext': Cython.Distutils.build_ext},
    ext_modules=[
        Extension('redshift.parser', ["redshift/parser.pyx"], language="c++",
                  include_dirs=[numpy.get_include(), pjoin(virtual_env, 'include')]),
        Extension('redshift._state', ["redshift/_state.pyx"],
                  include_dirs=[numpy.get_include(), os.path.join(virtual_env, 'include')]),
        Extension('redshift.io_parse', ["redshift/io_parse.pyx"], language="c++"),
        Extension('redshift.features', ["redshift/features.pyx"], language="c++"),
        Extension('svm.cy_svm', ['svm/cy_svm.pyx'], language="c++"),
        Extension('svm.cy_svm', ['svm/multitron.pyx'], language="c++",
                  include_dirs=[os.path.join('svm/ext', LIBLINEAR), numpy.get_include(), os.path.join(virtual_env, 'include')],
                  runtime_library_dirs=[os.path.join(pwd, 'lib')],
                  library_dirs=[os.path.join(pwd, 'lib')],
                  extra_objects=[os.path.join(pwd, 'lib', 'liblinear.so.1')])
        ]
    )



