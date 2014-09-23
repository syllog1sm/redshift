from __future__ import unicode_literals
import pytest

from redshift.pystate import PyState

@pytest.fixture
def string():
    return "This is a test ."

@pytest.fixture
def gold():
    return [(2, 1, b'SUBJ'), (2, 4, b'OBJ'), (4, 3, b'OTHER'), (2, 5, b'OTHER'),
            (6, 2, b'ROOT')]

def test_init_valid(string):
    state = PyState(string)
    assert state.is_valid('S')
    assert not state.is_valid('D')
    assert not state.is_valid('R-OBJ')
    assert not state.is_valid('L-SUBJ')
    
def test_right_valid(string):
    state = PyState(string, transitions=['S'])
    assert state.is_valid('R-OBJ')

def test_left_valid(string):
    state = PyState(string, transitions=['S'])
    assert state.is_valid('L-SUBJ')


def test_reduce_invalid(string):
    state = PyState(string, transitions=['S'])
    assert not state.is_valid('D')


def test_reduce_valid(string):
    state = PyState(string, transitions=['S', 'R-OBJ'])
    assert state.is_valid('D')


def test_left_invalid(string):
    state = PyState(string, transitions=['S', 'R-OBJ'])
    assert not state.is_valid('L-SUBJ')


def test_init_oracle(string, gold):
    state = PyState(string, gold=gold)
    assert state.is_gold('S')
    assert not state.is_gold('D')
    assert not state.is_gold('R-OBJ')
    assert not state.is_gold('L-SUBJ')


def test_gold_left_oracle(string, gold):
    state = PyState(string, gold=gold)
    state.transition('S')
    assert state.is_gold('L-SUBJ')
    assert not state.is_gold('L-OTHER')
    assert not state.is_gold('S')
    assert not state.is_gold('D')
    assert not state.is_gold('R-OBJ')
    state.transition('L-SUBJ')
    assert not state.is_valid('L-SUBJ')
    assert not state.is_valid('R-OTHER')
    assert not state.is_valid('D')
    assert state.is_gold('S')
    state.transition('S')
    state.transition('S')
    assert not state.is_gold('L-SUBJ')
    assert state.is_gold('L-OTHER')
    assert not state.is_gold('S')
    assert not state.is_gold('D')
    assert not state.is_gold('R-OBJ')
 

def test_gold_right_oracle(string, gold):
    state = PyState(string, gold=gold)
    state.transition('S')
    state.transition('L-SUBJ')
    state.transition('S')
    state.transition('S')
    state.transition('L-OTHER')
    assert not state.is_gold('R-OTHER')
    assert not state.is_gold('S')
    assert not state.is_gold('D')
    assert not state.is_gold('L-OTHER')
    assert state.is_gold('R-OBJ')
 

def test_gold_reduce_oracle(string, gold):
    state = PyState(string, gold=gold)
    state.transition('S')
    state.transition('L-SUBJ')
    state.transition('S')
    state.transition('S')
    state.transition('L-OTHER')
    state.transition('R-OBJ')
    assert not state.is_gold('R-OTHER')
    assert not state.is_gold('S')
    assert state.is_gold('D')
    assert not state.is_gold('L-ROOT')
    assert not state.is_gold('R-OBJ')
 

def test_gold_end_oracle(string, gold):
    state = PyState(string, gold=gold)
    state.transition('S')
    state.transition('L-SUBJ')
    state.transition('S')
    state.transition('S')
    state.transition('L-OTHER')
    state.transition('R-OBJ')
    state.transition('D')
    assert not state.is_gold('D')
    assert not state.is_gold('S')
    assert state.is_gold('R-OTHER')
    state.transition('R-OTHER')
    assert state.is_gold('D')
    state.transition('D')
    assert not state.is_valid('S')
    assert not state.is_valid('D')
    assert not state.is_valid('R-OTHER')
    assert state.is_gold('L-ROOT')
    assert not state.is_gold('L-OTHER')



# Now test left/right moves when we have "sunk cost" arcs
def test_sunk_left(string, gold):
    state = PyState(string, gold=gold)
    state.transition('S')
    state.transition('L-SUBJ')
    state.transition('S')
    state.transition('S')
    assert not state.is_gold('S')
    state.transition('S')
    # Now have (is, a, test) on stack --- this is wrong
    assert state.is_gold('L-SUBJ')
    assert state.is_gold('L-ROOT')
    assert state.is_gold('L-OTHER')
    assert not state.is_gold('R-OTHER')
    assert not state.is_gold('R-OBJ')
    assert not state.is_gold('D')
    assert not state.is_gold('S')
    state.transition('L-SUBJ')
    assert state.is_gold('L-SUBJ')
    assert state.is_gold('L-ROOT')
    assert state.is_gold('L-OTHER')
    state.transition('L-ROOT')
    assert state.top == 2
    assert state.is_gold('R-OTHER')


def test_sunk_right(string, gold):
    state = PyState(string, gold=gold)
    state.transition('S')
    assert state.top == 1
    assert not state.is_gold('R-OBJ')
    assert state.top == 1
    state.transition('R-OBJ')
    assert state.top == 2
    assert not state.is_gold('D')
    state.transition('D')
    assert not state.is_gold('R-OBJ')
    state.transition('R-OBJ')
    assert state.is_gold('D')
    assert not state.is_gold('L-OTHER')
    assert not state.is_valid('L-OTHER')
