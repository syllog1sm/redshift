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

::

    >>> import redshift.parser
    >>> parser = redshift.parser.load_parser('/tmp/stanford_beam8')
    >>> import redshift.io_parse
    >>> sentences = redshift.io_parse.read_pos('Barry/NNP Wright/NNP ,/, acquired/VBN by/IN Applied/NNP for/IN $/$ 147/CD million/CD ,/, makes/VBZ computer-room/JJ equipment/NN and/CC vibration-control/JJ systems/NNS ./.')
    >>> parser.add_parses(sentences)
    >>> import sys; sentences.write_parses(sys.stdout)
    0       Barry   NNP     1       nn
    1       Wright  NNP     11      nsubj
    2       ,       ,       1       P
    3       acquired        VBN     1       partmod
    4       by      IN      3       prep
    5       Applied NNP     4       pobj
    6       for     IN      3       prep
    7       $       $       6       pobj
    8       147     CD      7       number
    9       million CD      7       number
    10      ,       ,       1       P
    11      makes   VBZ     -1      ROOT
    12      computer-room   JJ      13      amod
    13      equipment       NN      11      dobj
    14      and     CC      13      cc
    15      vibration-control       JJ      16      amod
    16      systems NNS     13      conj
    17      .       .       11      P

The command-line interfaces have a lot of probably-confusing options for my current research. The main scripts I use are
scripts/train.py, scripts/parse.py, and scripts/evaluate.py . All print usage information, and require the plac library.


The following commands will set up a virtualenv with Python 2.7.5, the parser, and its core dependencies from scratch::


From a Unix/OSX terminal, after compilation, and within the "redshift" directory:

::

    $ export PYTHONPATH=`pwd`
    $ ./scripts/train.py # Use -h or --help for more detailed info. Most of these are research flags.
    usage: train.py [-h] [-a static] [-i 15] [-k 1] [-f 10] [-r] [-d] [-u] [-n 0] [-s 0] train_loc model_loc
    train.py: error: too few arguments
    $ ./scripts/train.py -k 16 -p <CoNLL formatted training data> <output model directory>
    $ ./scripts/parse.py <model directory produced by train.py> <input> <output_dir>
    $ ./scripts/evaluate.py output_dir/parses <gold file>
    
In more detail:

* Ensure your PYTHONPATH variable includes the redshift directory
* Most of the training-script flags refer to research settings.
* the k parameter controls the speed-accuracy trade-off, via the beam-width. Run-time is roughly O(nk), where n is the number of words, and k is the beam-width. In practice it's slightly sub-linear in k due to some simple memoisation. Accuracy plateaus at about k=64. For k=1, use "-a dyn -r -d", to enable some recent special-case wizardry that gives the k=1 case over 1% extra accuracy, at no run-time cost.
* The -p flag tells train.py to train a POS tagger.
* parse.py reads in the training configuration from "parser.cfg", which sits in the output model directory.
* The parser currently expects one sentence per line, space-separated tokens, tokens of the form word/POS.
* evaluate.py runs as a separate script from parse.py so that the parser never sees the answers, and cannot "accidentally cheat".

Installation
------------

The following commands will set up a virtualenv with Python 2.7.5, the parser, and its core dependencies from scratch::

    $ git clone https://github.com/syllog1sm/redshift.git
    $ cd redshift
    $ ./make_virtualenv.sh # Downloads Python 2.7.5 and virtualenv
    $ source $HOME/rsve/bin/activate
    $ ./install_sparsehash.sh # Downloads the Google sparsehash 2.2 library and installs it under the virtualenv
    $ pip install cython
    $ python setup.py build_ext --inplace # site-install currently broken, use --inplace
    $ export PYTHONPATH=`pwd`:$PYTHONPATH # ...and set PYTHONPATH.
    $ pip install plac # For command-line interfaces

virtualenv is not a requirement, although it's useful.  If a virtualenv is not active (i.e. if the $VIRTUALENV
environment variable is not set), install_sparsehash.sh will install the Google sparsehash library under redshift/ext/,
to avoid assuming root privileges for the installation.  To install sparsehash elsewhere, add the path to the "includes"
list in setup.py

You might wish to handle the tasks covered by ./make_virtualenv.sh and ./install_sparsehash.sh yourself, depending on
how you want your environment set up.

Cython
------

redshift is written almost entirely in Cython, a superset of the Python language that additionally supports
calling C/C++ functions and declaring C/C++ types on variables and class attributes. This allows the compiler to
generate very efficient C/C++ code from Cython code. Many popular Python packages, such as numpy, scipy and lxml,
rely heavily on Cython code.

A Cython source file such as learn/perceptron.pyx is compiled into learn/perceptron.cpp and learn/perceptron.so by
the project's setup.py file. The module can then by imported by standard Python code, although only the pure-Python
functions (declared by "def", instead of "cdef") will be accessible.

The parser currently has Cython as a requirement, instead of distributing
the "compiled" .cpp files as part of the release (against Cython's recommendation). This could change in future,
but currently it feels strange to have a "source" release that users wouldn't be able to modify. 

<<<<<<< HEAD:README.md
=======
LICENSE
---------------

I'm still working out how to specify the license, but my intention at the moment is:

- FOSS for non-commercial use
- Modifications should be distributed
- Commercial use licenses available on request. These will be granted pretty much automatically to any company that isn't yet profitable, or really anyone who isn't big.
- RESTful parser APIs to make it easier to start using the parser.
    
::

    Copyright (C) 2013 Matthew Honnibal
>>>>>>> origin:README.rst
