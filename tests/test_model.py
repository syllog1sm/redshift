from __future__ import division

import pytest

from learn.thinc import LinearModel

def test_basic():
    model = LinearModel(5)
    model.update({0: {1: 1, 3: -5}, 1: {2: 5, 3: 5}})
    s1 = model([1])
    assert s1[0] == 1
    assert s1[1] == 0
    s2 = model([2])
    assert s2[0] == 0
    assert s2[1] == 5
    s3 = model([3])
    assert s3[0] == -5
    assert s3[1] == 5
    scores = model([1, 2, 3])
    assert scores[0] == sum((s1[0], s2[0], s3[0]))
    assert scores[1] == sum((s1[1], s2[1], s3[1]))


@pytest.fixture
def model():
    m = LinearModel(3)
    classes = range(3)
    instances = [
        {
            0: {1: -1, 2: 1},
            1: {1: 5, 2: -5},
            2: {1: 3, 2: -3},
        },
        {
            0: {1: -1, 2: 1},
            1: {1: -1, 2: 2},
            2: {1: 3, 2: -3},
        },
        {
            0: {1: -1, 2: 2},
            1: {1: 5, 2: -5}, 
            2: {4: 1, 5: -7, 2: 1}
        }
    ]

    for counts in instances:
        m.update(counts)
    return m

def test_averaging(model):
    model.end_training()
    # Feature 1
    assert model([1])[0] == sum([-1, -2, -3]) / 3
    assert model([1])[1] == sum([5, 4, 9]) / 3
    assert model([1])[2] == sum([3, 6, 6]) / 3
    # Feature 2
    assert model([2])[0] == sum([1, 2, 4]) / 3
    assert model([2])[1] == sum([-5, -3, -8]) / 3
    assert model([2])[2] == sum([-3, -6, -5]) / 3
    # Feature 3 (absent)
    assert model([3])[0] == 0
    assert model([3])[1] == 0
    assert model([3])[2] == 0
    # Feature 4
    assert model([4])[0] == sum([0, 0, 0]) / 3
    assert model([4])[1] == sum([0, 0, 0]) / 3
    assert model([4])[2] == sum([0, 0, 1]) / 3
    # Feature 5
    assert model([5])[0] == sum([0, 0, 0]) / 3
    assert model([5])[1] == sum([0, 0, 0]) / 3
    assert model([5])[2] == sum([0, 0, -7]) / 3






def test_dump():
    pass


def test_load():
    pass
