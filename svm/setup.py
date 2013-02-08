#!/usr/bin/env python

from distutils.core import setup
import os
import os.path
import numpy

from Cython.Distutils.extension import Extension
import Cython.Distutils

LIBLINEAR = 'liblinear-weights-1.91'

pwd = os.path.dirname(__file__)
virtual_env = os.environ.get('VIRTUAL_ENV', '')

setup(
  name = 'Cython bindings for liblinear/libsvm',
  cmdclass = {'build_ext': Cython.Distutils.build_ext},
  ext_modules = [
        Extension(
            "cy_svm",                 # name of extension
            ["cy_svm.pyx"],           # filename of our Pyrex/Cython source

            language="c++",              # this causes Pyrex/Cython to create C++ source
            libraries=['linear'],
            include_dirs=[os.path.join('ext', LIBLINEAR), numpy.get_include(), os.path.join(virtual_env, 'include')],
            runtime_library_dirs=[os.path.join(pwd, 'lib')],
            library_dirs=[os.path.join(pwd, 'lib')],
            extra_objects=[os.path.join(pwd, 'lib', 'liblinear.so.1')]),

        Extension(
            "multitron",
            ["multitron.pyx"],
            include_dirs=[numpy.get_include(), os.path.join(virtual_env, 'include')],
            language="c++")
        ]
)


