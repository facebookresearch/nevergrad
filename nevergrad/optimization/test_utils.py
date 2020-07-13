# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import pytest
import numpy as np
import nevergrad as ng
from nevergrad.common import testing
from nevergrad.parametrization import parameter as p
from .test_base import CounterFunction
from . import experimentalvariants as xpvariants
from . import utils


def test_value_and_point() -> None:
    param = ng.p.Scalar(init=12.0)
    v = utils.MultiValue(param, 4, reference=param)
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
    assert v.parameter.value == 12


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
    param = zeroptim.parametrization
    for k in range(4):
        array = (float(k),)
        zeroptim.archive[array] = utils.MultiValue(param, k, reference=param)
        zeroptim.archive[array].count += (4 - k)
    nash = utils._get_nash(zeroptim)
    testing.printed_assert_equal(nash, [((2,), 3), ((1,), 4), ((0,), 5)])
    np.random.seed(12)
    output = utils.sample_nash(zeroptim)
    np.testing.assert_equal(output, (2,))


def test_archive() -> None:
    data = [1, 4.5, 12, 0]
    archive: utils.Archive[int] = utils.Archive()
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
    archive: utils.Archive[float] = utils.Archive()
    archive[[12, 0.]] = 12.
    np.testing.assert_raises(AssertionError, archive.__getitem__, [12, 0])  # int instead of float
    np.testing.assert_raises(AssertionError, archive.__getitem__, [[12], [0.]])  # int instead of float
    np.testing.assert_raises(RuntimeError, archive.keys)
    np.testing.assert_raises(RuntimeError, archive.items)


def test_pruning() -> None:
    param = ng.p.Scalar(init=12.0)
    archive: utils.Archive[utils.MultiValue] = utils.Archive()
    for k in range(3):
        value = utils.MultiValue(param, float(k), reference=param)
        archive[(float(k),)] = value
    value = utils.MultiValue(param, 1., reference=param)
    value.add_evaluation(1.)
    archive[(3.,)] = value
    # pruning
    pruning = utils.Pruning(min_len=1, max_len=3)
    # 0 is best optimistic and average, and 3 is best pessimistic (variance=0)
    archive = pruning(archive)
    testing.assert_set_equal([x[0] for x in archive.keys_as_arrays()], [0, 3], err_msg=f"Repetition #{k+1}")
    pickle.dumps(archive)  # should be picklable
    # should not change anything this time
    archive2 = pruning(archive)
    testing.assert_set_equal([x[0] for x in archive2.keys_as_arrays()], [0, 3], err_msg=f"Repetition #{k+1}")


@pytest.mark.parametrize("nw,dimension,expected_min,expected_max", [  # type: ignore
    (12, 8, 100, 1000),
    (24, 8, 168, 1680),
    (24, 100000, 168, 671),
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
    # pickling
    string = pickle.dumps(uidq)
    pickle.loads(string)
    # clearing
    uidq.clear()
    with pytest.raises(RuntimeError):
        uidq.ask()


def test_bound_scaler() -> None:
    ref = p.Instrumentation(
        p.Array(shape=(1, 2)).set_bounds(-12, 12, method="arctan"),
        p.Array(shape=(2,)).set_bounds(-12, 12, full_range_sampling=False),
        lr=p.Log(lower=0.001, upper=1000),
        stuff=p.Scalar(lower=-1, upper=2),
        unbounded=p.Scalar(lower=-1, init=0.0),
        value=p.Scalar(),
        letter=p.Choice("abc"),
    )
    param = ref.spawn_child()
    scaler = utils.BoundScaler(param)
    output = scaler.transform([1.0] * param.dimension, lambda x: x)
    param.set_standardized_data(output)
    (array1, array2), values = param.value
    np.testing.assert_array_almost_equal(array1, [[12, 12]])
    np.testing.assert_array_almost_equal(array2, [1, 1])
    assert values["stuff"] == 2
    assert values["unbounded"] == 1
    assert values["value"] == 1
    np.testing.assert_almost_equal(values["lr"], 1000)
    # again, on the middle point
    output = scaler.transform([0] * param.dimension, lambda x: x)
    param.set_standardized_data(output)
    np.testing.assert_almost_equal(param.value[1]["lr"], 1.0)
    np.testing.assert_almost_equal(param.value[1]["stuff"], 0.5)
