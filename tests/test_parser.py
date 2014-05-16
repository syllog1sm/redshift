"""
Test main parser API
"""
import os.path

import pytest


def local_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


model_dir = local_path('model')
train_str = open(local_path('train.10.conll')).read()


@pytest.fixture
def train_dir():
    import redshift.parser
    redshift.parser.train(train_str, model_dir)
    return model_dir


@pytest.fixture
def parser(train_dir):
    import redshift.parser
    return redshift.parser.Parser(train_dir)


@pytest.fixture
def sentence():
    from redshift.sentence import Input
    return Input.from_pos('This/?? is/?? a/?? test/?? ./.')


def test_parse(parser, sentence):
    parser.parse(sentence)
    assert sentence.length == 7
    tokens = list(sentence.tokens)
    assert tokens[0].word == 'This'
    assert tokens[1].word == 'is'
    assert tokens[2].word == 'a'
    assert tokens[3].word == 'test'
    assert tokens[4].word == '.'
