# Redshift #

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
* redshift.parser.GreedyParser adds the non-monotonic model of Honnibal et al (2013) to the dynamic oracle
model of Goldberg and Nivre (2012)
* redshift.features includes the standard Zhang and Nivre (2011) feature set, and also some work pending publication.

# Example usage #

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

Published results always refer to multiple runs (usually with 20 random seeds). These experiments are automated via fabric,
which I also usually use for compilation (e.g. "fab make").

INSTALLATION

The following commands will set up a virtualenv with Python 2.7.5, the parser, and its core dependencies from scratch:

    git clone https://github.com/syllog1sm/redshift.git
    cd redshift
    ./make_virtualenv.sh
    source $HOME/rsve/bin/activate
    ./install_sparsehash.sh
    pip install cython
    python setup.py build_ext --inplace
    export PYTHONPATH=`pwd`:$PYTHONPATH

virtualenv is not a requirement, although it's useful.  If a virtualenv is not active (i.e. if the $VIRTUALENV
environment variable is not set), install_sparsehash.sh will install the Google sparsehash library under redshift/ext/,
to avoid assuming root privileges for the installation.  To install sparsehash elsewhere, add the path to the "includes"
list in setup.py

You might wish to handle the tasks covered by ./make_virtualenv.sh and ./install_sparsehash.sh yourself, depending on
how you want your environment set up. The parser currently has cython as a requirement, instead of distributing
the "compiled" .cpp files as part of the release (against Cython's recommendation). This will probably change in future.

To use the command line scripts and all auxiliary gadgets, you'll also need:

    pip install plac
    pip install fabric

# LICENSE (GPL 3) #

    Copyright (C) 2013 Matthew Honnibal

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
