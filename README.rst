Redshift
========

Redshift is a natural-language syntactic dependency parser.  The current release features fast and accurate parsing,
but requires the text to be pre-processed.  Future releases will integrate tokenisation and part-of-speech tagging,
and have special features for parsing informal text.

If you don't know what a syntactic dependency is, read this:
http://googleresearch.blogspot.com.au/2013/05/syntactic-ngrams-over-time.html

Main features:

* Fast linear time parsing: the slowest model is still over 100 sentences/second
* State-of-the-art accuracy: 93.5% UAS on English (Stanford scheme, WSJ 23)
* Super fast "greedy" mode: over 1,000 sentences per second at 91.5% accuracy
* Native Python interface (the parser is written in Cython)

Key techniques:

* Arc-eager transition-based dependency parser
* Averaged perceptron for learning
* redshift.parser.BeamParser is basically the model of Zhang and Nivre (2011)
* redshift.parser.GreedyParser adds the non-monotonic model of Honnibal et al (2013) to the dynamic oracle model of Goldberg and Nivre (2012)
* redshift.features includes the standard Zhang and Nivre (2011) feature set, and also some work pending publication.

Example usage
-------------

Here is an example of how the parser is called from Python, once you have a model trained:

::

    >>> import redshift.parser
    >>> from redshift.sentence import Input
    >>> parser = redshift.parser.Parser(<model directory>)
    >>> sentence = Input.from_untagged(['A', 'list', 'of', 'tokens', 'is', 'required', '.'])
    >>> parser.parse(sentence)
    >>> print sentence.to_conll()

The command-line interfaces have a lot of probably-confusing options for my current research. The main scripts I use are
scripts/train.py, scripts/parse.py, and scripts/evaluate.py . All print usage information, and require the plac library.


From a Unix/OSX terminal, after compilation, and within the "redshift" directory:

::

    $ export PYTHONPATH=`pwd`
    $ ./scripts/train.py # Use -h or --help for more detailed info. Most of these are research flags.
    usage: train.py [-h] [-a static] [-i 15] [-k 1] [-f 10] [-r] [-d] [-u] [-n 0] [-s 0] train_loc model_loc
    train.py: error: too few arguments
    $ ./scripts/train.py -k 16  <CoNLL formatted training data> <output model directory>
    $ ./scripts/parse.py <model directory produced by train.py> <input> <output_dir>
    $ ./scripts/evaluate.py output_dir/parses <gold file>
    
In more detail:

* Ensure your PYTHONPATH variable includes the redshift directory
* Most of the training-script flags refer to research settings.
* the k parameter controls the speed-accuracy trade-off, via the beam-width. Run-time is roughly O(nk), where n is the number of words, and k is the beam-width. In practice it's slightly sub-linear in k due to some simple memoisation. Accuracy plateaus at about k=64. For k=1, use "-a dyn -r -d", to enable some recent special-case wizardry that gives the k=1 case over 1% extra accuracy, at no run-time cost.
* parse.py reads in the training configuration from "parser.cfg", which sits in the output model directory.
* The parser currently expects one sentence per line, space-separated tokens, tokens of the form word/POS.
* evaluate.py runs as a separate script from parse.py so that the parser never sees the answers, and cannot "accidentally cheat".

Installation
------------

The following commands will set up a virtualenv with Python 2.7.5, the parser, and its core dependencies from scratch::

    $ git clone https://github.com/syllog1sm/redshift.git
    $ cd redshift
    $ ./make_virtualenv.sh # Downloads Python 2.7.5 and virtualenv
    $ source .env/bin/activate
    $ pip install distribute
    $ pip install cython
    $ pip install -r requirements.txt
    $ export PYTHONPATH=`pwd`:$PYTHONPATH # ...and set PYTHONPATH.
    $ fab make test

The make_virtualenv.sh script downloads and compiles Python 2.7.5, and uses it to create a virtualenv. This is one way to use a version of Python that isn't system-wide, or to control the compiler that Cython will use.  You may not need to do this, or you may wish to do it manually --- it's up to you.

virtualenv is not a requirement, although it's useful.  If a virtualenv is not active (i.e. if the $VIRTUALENV
environment variable is not set), you'll need to ensure that the setup.py file knows where to find the C headers that the murmurhash dependency installs.

Cython
------

redshift is written almost entirely in Cython, a superset of the Python language that additionally supports
calling C/C++ functions and declaring C/C++ types on variables and class attributes. This allows the compiler to
generate very efficient C/C++ code from Cython code. Many popular Python packages, such as numpy, scipy and lxml,
rely heavily on Cython code.

A Cython source file such as redshift/parser.pyx is compiled into redshift/parser.cpp and redshift/parser.so by
the project's setup.py file. The module can then by imported by standard Python code, although only the pure-Python
functions (declared by "def" and "cpdef", instead of "cdef") will be accessible.

The parser currently has Cython as a requirement, instead of distributing
the "compiled" .cpp files as part of the release (against Cython's recommendation). This could change in future,
but currently it feels strange to have a "source" release that users wouldn't be able to modify. 

LICENSE
---------------

I'm still working out how to specify the license, but my intention at the moment is:

- FOSS for non-commercial use
- Modifications should be distributed
- Commercial use licenses available on request. These will be granted pretty much automatically to any company that isn't yet profitable, or really anyone who isn't big.
- RESTful parser APIs to make it easier to start using the parser.
    
::

    Copyright (C) 2014 Matthew Honnibal
