# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
from ..common import testing
from .test_base import CounterFunction
from .optimizerlib import Zero
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
    zeroptim = Zero(instrumentation=1, budget=4, num_workers=1)
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
    items = list(archive.items_as_array())
    assert isinstance(items[0][0], np.ndarray)
    keys = list(archive.keys_as_array())
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


class Partitest(utils.Particle):

    def __init__(self, number: int) -> None:
        super().__init__()
        self.number = number


def test_population_queue() -> None:
    particles = [Partitest(k) for k in range(4)]
    pop = utils.Population(particles[2:])
    pop.extend(particles[:2])  # should append queue on the left
    p = pop.get_queued()
    assert p.number == 0
    nums = [pop.get_queued(remove=True).number for _ in range(4)]
    np.testing.assert_equal(nums, [0, 1, 2, 3])
    np.testing.assert_raises(RuntimeError, pop.get_queued)  # nothing more in queue
    pop.set_queued(particles[1])
    p = pop.get_queued()
    assert p.number == 1
    np.testing.assert_raises(ValueError, pop.set_queued, Partitest(5))  # not in pop


def test_population_link() -> None:
    particles = [Partitest(k) for k in range(4)]
    pop = utils.Population(particles)
    np.testing.assert_raises(ValueError, pop.set_linked, "blublu", Partitest(5))  # not in pop
    p = particles[0]
    pop.set_linked(12, p)
    p2 = pop.get_linked(12)
    assert p2.uuid == p.uuid
    pop.del_link(12, p)
    np.testing.assert_raises(KeyError, pop.get_linked, 12)  # removed


def test_population_replace() -> None:
    particles = [Partitest(k) for k in range(4)]
    pop = utils.Population(particles)
    pop.set_linked(12, particles[2])
    key = pop.replace(particles[2], Partitest(5))
    assert key == 12
    assert pop.get_queued().number == 5
    for uuid in pop.uuids:
        # checks that it exists and correctly linked
        pop[uuid]  # pylint: disable= pointless-statement


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
    for k in range(2):
        archive = pruning(archive)
        testing.assert_set_equal([x[0] for x in archive.keys_as_array()], [0, 3], err_msg=f"Repetition #{k+1}")


@pytest.mark.parametrize("dimension,expected_max", [(100, 1342177), (10000, 13421), (1000000, 1080)])  # type: ignore
def test_pruning_sensible_default(dimension: int, expected_max: int) -> None:
    pruning = utils.Pruning.sensible_default(num_workers=12, dimension=dimension)
    assert pruning.min_len == 36
    assert pruning.max_len == expected_max
