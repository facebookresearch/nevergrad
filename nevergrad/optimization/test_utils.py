# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
from ..common import testing
from .test_base import CounterFunction
from . import experimentalvariants as xpvariants
from . import utils


def test_value_and_point() -> None:
    v = utils.Value(4)
    np.testing.assert_equal(v.count, 1)
    v.add_evaluation(3)
    np.testing.assert_equal(v.count, 2)
    np.testing.assert_equal(v.mean, 3.5)
    np.testing.assert_equal(v.square, 12.5)
    np.testing.assert_almost_equal(v.variance, 0.3536, decimal=4)
    assert v.optimistic_confidence_bound < v.pessimistic_confidence_bound
    assert v.get_estimation("optimistic") < v.get_estimation("pessimistic")
    np.testing.assert_raises(NotImplementedError, v.get_estimation, "blublu")
    repr(v)
    # now test point based on this value
    p = utils.Point(np.array([0., 0]), v)
    np.testing.assert_equal(p.mean, 3.5)
    np.testing.assert_almost_equal(p.variance, 0.3536, decimal=4)
    repr(p)
    np.testing.assert_raises(AssertionError, utils.Point, (0, 0), 3)


def test_sequential_executor() -> None:
    func = CounterFunction()
    executor = utils.SequentialExecutor()
    job1 = executor.submit(func, [3])
    np.testing.assert_equal(job1.done(), True)
    np.testing.assert_equal(job1.result(), 4)
    np.testing.assert_equal(func.count, 1)
    job2 = executor.submit(func, [3])
    np.testing.assert_equal(job2.done(), True)
    np.testing.assert_equal(func.count, 1)  # not computed just yet
    job2.result()
    np.testing.assert_equal(func.count, 2)


def test_get_nash() -> None:
    zeroptim = xpvariants.Zero(parametrization=1, budget=4, num_workers=1)
    for k in range(4):
        array = (float(k),)
        zeroptim.archive[array] = utils.Value(k)
        zeroptim.archive[array].count += (4 - k)
    nash = utils._get_nash(zeroptim)
    testing.printed_assert_equal(nash, [((2,), 3), ((1,), 4), ((0,), 5)])
    np.random.seed(12)
    output = utils.sample_nash(zeroptim)
    np.testing.assert_equal(output, (2,))


def test_archive() -> None:
    data = [1, 4.5, 12, 0]
    archive = utils.Archive[int]()
    archive[np.array(data)] = 12
    np.testing.assert_equal(archive[np.array(data)], 12)
    np.testing.assert_equal(archive.get(data), 12)
    np.testing.assert_equal(archive.get([0, 12.]), None)
    y = np.frombuffer(next(iter(archive.bytesdict.keys())))
    assert data in archive
    np.testing.assert_equal(y, data)
    items = list(archive.items_as_arrays())
    assert isinstance(items[0][0], np.ndarray)
    keys = list(archive.keys_as_arrays())
    assert isinstance(keys[0], np.ndarray)
    repr(archive)
    str(archive)


def test_archive_errors() -> None:
    archive = utils.Archive[float]()
    archive[[12, 0.]] = 12.
    np.testing.assert_raises(AssertionError, archive.__getitem__, [12, 0])  # int instead of float
    np.testing.assert_raises(AssertionError, archive.__getitem__, [[12], [0.]])  # int instead of float
    np.testing.assert_raises(RuntimeError, archive.keys)
    np.testing.assert_raises(RuntimeError, archive.items)


def test_pruning() -> None:
    archive = utils.Archive[utils.Value]()
    for k in range(3):
        value = utils.Value(float(k))
        archive[(float(k),)] = value
    value = utils.Value(1.)
    value.add_evaluation(1.)
    archive[(3.,)] = value
    # pruning
    pruning = utils.Pruning(min_len=1, max_len=3)
    # 0 is best optimistic and average, and 3 is best pessimistic (variance=0)
    archive = pruning(archive)
    testing.assert_set_equal([x[0] for x in archive.keys_as_arrays()], [0, 3], err_msg=f"Repetition #{k+1}")
    # should not change anything this time
    archive = pruning(archive)
    testing.assert_set_equal([x[0] for x in archive.keys_as_arrays()], [0, 3], err_msg=f"Repetition #{k+1}")


@pytest.mark.parametrize("nw,dimension,expected_min,expected_max", [  # type: ignore
    (12, 8, 100, 1000),
    (24, 8, 168, 1680),
    (24, 200000, 168, 671),
    (24, 1000000, 168, 504),
])
def test_pruning_sensible_default(nw: int, dimension: int, expected_min: int, expected_max: int) -> None:
    pruning = utils.Pruning.sensible_default(num_workers=nw, dimension=dimension)
    assert pruning.min_len == expected_min
    assert pruning.max_len == expected_max


def test_uid_queue() -> None:
    uidq = utils.UidQueue()
    for uid in ["a", "b", "c"]:
        uidq.tell(uid)
    for uid in ["a", "b"]:
        assert uidq.ask() == uid
    uidq.tell("b")
    for uid in ["c", "b", "a", "c", "b", "a"]:
        assert uidq.ask() == uid
    # discarding (in asked, and in told)
    uidq.discard("b")
    for uid in ["c", "a", "c", "a"]:
        assert uidq.ask() == uid
    uidq.tell("a")
    uidq.discard("a")
    for uid in ["c", "c"]:
        assert uidq.ask() == uid
    # clearing
    uidq.clear()
    with pytest.raises(RuntimeError):
        uidq.ask()
