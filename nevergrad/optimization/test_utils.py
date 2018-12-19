# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    p = utils.Point((0, 0), v)
    np.testing.assert_equal(p.mean, 3.5)
    np.testing.assert_almost_equal(p.variance, 0.3536, decimal=4)
    repr(p)
    np.testing.assert_raises(AssertionError, utils.Point, (0, 0), 3)


def test_sequential_executor() -> None:
    func = CounterFunction()
    executor = utils.SequentialExecutor()
    job1 = executor.submit(func, 3)
    np.testing.assert_equal(job1.done(), True)
    np.testing.assert_equal(job1.result(), 4)
    np.testing.assert_equal(func.count, 1)
    executor.submit(func, 3)
    np.testing.assert_equal(func.count, 2)


def test_get_nash() -> None:
    zeroptim = Zero(dimension=1, budget=4, num_workers=1)
    for k in range(4):
        zeroptim.archive[(k,)] = utils.Value(k)
        zeroptim.archive[(k,)].count += (4 - k)
    nash = utils._get_nash(zeroptim)
    testing.printed_assert_equal(nash, [((2,), 3), ((1,), 4), ((0,), 5)])
    np.random.seed(12)
    output = utils.sample_nash(zeroptim)
    np.testing.assert_equal(output, (2,))
