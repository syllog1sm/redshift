from __future__ import unicode_literals
import pytest

from redshift.pystate import PyState

@pytest.fixture
def string():
    return "pass the pepper uh salt"

@pytest.fixture
def gold():
    return [(1, 5, b'OBJ'), (3, 3, b'ERASE'), (4, 4, b'FILL'), (5, 2, b'DET'),
            (6, 1, b'ROOT')]


@pytest.fixture
def state(string, gold):
    return PyState(string, gold=gold, left_labels=[b'DET', b'ROOT', b'OTHER'],
                   right_labels=[b'OBJ', b'OTHER'], dfl_labels=[b'FILL', b'ERASE'])


def test_init(state):
    assert state.is_gold('S')
    assert not state.is_gold('D')
    assert not state.is_gold('L-ROOT')
    assert not state.is_gold('R-OBJ')
    assert not state.is_gold('E-ERASE')


def test_bad_edit(state):
    state.transition('S')
    assert not state.is_gold('E-ERASE')
    assert not state.is_gold('E-FILL')
    

def test_non_monotonic_oracle(state):
    state.transition('S')
    state.transition('S')
    assert not state.is_gold('E-ERASE')
    assert not state.is_gold('E-FILL')
    assert state.is_gold('S')
    assert state.is_gold('L-DET')
    assert state.is_gold('L-OTHER')
    assert state.is_gold('L-ROOT')
    assert state.is_gold('R-OBJ') 
    assert state.is_gold('R-OTHER') 


def test_non_monotonic_right(state):
    state.transition('S')
    state.transition('S')
    assert state.is_gold('R-OBJ') 
    state.transition('R-OBJ')
    assert not state.is_gold('D')
    assert not state.is_gold('L-OTHER')
    assert state.is_gold('E-ERASE')
    assert not state.is_gold('E-FILL')


def test_non_monotonic_left(state):
    state.transition('S')
    state.transition('S')
    state.transition('L-DET')
    state.transition('S')
    assert state.is_gold('E-ERASE')
    assert not state.is_gold('E-FILL')
    assert not state.is_gold('L-DET')
    assert not state.is_gold('L-OTHER')
    assert not state.is_gold('L-ROOT')

