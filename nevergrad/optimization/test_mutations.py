# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.common import testing
from . import utils
from .mutations import Mutator
from .differentialevolution import Crossover


def test_significantly_mutate() -> None:
    rng = np.random.RandomState(12)
    output = Mutator(rng).significantly_mutate(0.3, 2)
    np.testing.assert_almost_equal(output, -0.68142, decimal=4)
    output = Mutator(rng).significantly_mutate(0.3, 2)
    np.testing.assert_almost_equal(output, -1.70073, decimal=4)
    output = Mutator(rng).significantly_mutate(0.3, 3)
    np.testing.assert_almost_equal(output, 0.75314, decimal=4)
    for _ in range(10):
        output = Mutator(rng).significantly_mutate(0.1, 2)
        np.testing.assert_array_less([output], [0.0])
        output = Mutator(rng).significantly_mutate(output, 2)
        np.testing.assert_array_less([0.0], [output])


def test_discrete_mutation() -> None:
    data = [0.1, -0.1, 1]
    rng = np.random.RandomState(12)
    output = Mutator(rng).discrete_mutation(data)
    np.testing.assert_almost_equal(output, [-0.33, -0.1, -0.42], decimal=2)


def test_crossover() -> None:
    data = [0.1, -0.1, 1, -0.1]
    rng = np.random.RandomState(15)
    output = Mutator(rng).crossover(data, 2 * np.array(data))
    np.testing.assert_almost_equal(output, [-1.1, -0.1, 2, -0.1], decimal=2)


@testing.parametrized(
    dicrete=("discrete_mutation",),
    portfolio_discrete=("portfolio_discrete_mutation",),
    doubledoerr=("doubledoerr_discrete_mutation",),
    doerr=("doerr_discrete_mutation",),
)
def test_run_with_array(name: str) -> None:
    data = [0.1, -0.1, 1, -0.2] * 3
    mutator = Mutator(np.random.RandomState(12))
    func = getattr(mutator, name)
    output1 = func(data)
    mutator.random_state.seed(12)
    output2 = func(np.array(data))
    np.testing.assert_equal(output1, output2)


@testing.parametrized(
    only_2=(2, 1.5),
    all_4=(4, 0.5),
)
def test_get_roulette(num: int, expected: str) -> None:
    param = ng.p.Scalar(init=12.0)
    rng = np.random.RandomState(24)
    archive: utils.Archive[utils.MultiValue] = utils.Archive()
    for k in range(4):
        archive[np.array([k + 0.5])] = utils.MultiValue(param, k, reference=param)
    output = Mutator(rng).get_roulette(archive, num)
    np.testing.assert_equal(output, expected)


@testing.parametrized(
    cr_0=(0.0, 24, [0, 0, 3, 0]),  # one item is always kept
    cr_1=(1.0, 24, [1, 2, 3, 4]),
    cr_02=(0.2, 24, [0, 2, 3, 0]),
    onepoint=("onepoint", 24, [0, 0, 0, 4, 5, 6]),
    twopoints=("twopoints", 16, [0, 0, 3, 4, 0, 0]),
    twopoints_special=("twopoints", 20, [0, 0, 0, 0, 5, 6]),  # redraws since bounds was [0, 6]
)
def test_de_crossover(crossover_param: tp.Union[str, float], seed: int, expected: tp.List[int]) -> None:
    rng = np.random.RandomState(seed)
    crossover = Crossover(rng, crossover_param)
    donor = np.arange(1, len(expected) + 1)
    crossover.apply(donor, 0.0 * donor)
    np.testing.assert_array_equal(donor, expected)
