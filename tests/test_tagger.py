"""
Test POS tagger
"""
import os.path
import pytest


def local_path(filename):
    return os.path.join(os.path.dirname(__file__), filename)


model_dir = local_path('model')
train_str = open(local_path('train.10.conll')).read()


@pytest.fixture
def train_dir():
    import redshift.tagger
    sent_strs = []
    for sent_str in train_str.strip().split('\n\n'):
        sent = []
        for tok_str in sent_str.strip().split('\n'):
            fields = tok_str.split()
            sent.append('%s/%s' % (fields[1], fields[3]))
        sent_strs.append(' '.join(sent))
    train_pos = '\n'.join(sent_strs)
    redshift.tagger.train(train_pos, model_dir)
    return model_dir


@pytest.fixture
def tagger(train_dir):
    import redshift.tagger
    return redshift.tagger.Tagger(train_dir)


@pytest.fixture
def sentence():
    from redshift.sentence import Input
    return Input.from_pos('This/?? is/?? a/?? test/?? ./.')


def test_feature_thresh():
    # assert that we end up with fewer features by setting a feature threshold
    pass

def test_beam_width():
    # assert that accuracy is higher with beam-width 4 than beam-width 1.
    pass

def test_feature_set():
    # assert that two different debug feature sets change the number of features.
    pass

def test_tag(tagger, sentence):
    tagger.tag(sentence)
    assert sentence.length == 7
    tokens = list(sentence.tokens)
    assert tokens[0].word == 'This'
    assert tokens[1].word == 'is'
    assert tokens[2].word == 'a'
    assert tokens[3].word == 'test'
    assert tokens[4].word == '.'
