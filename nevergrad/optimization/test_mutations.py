# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Any
import numpy as np
from ..common import testing
from . import utils
from . import mutations


def test_discrete_mutation() -> None:
    data = [0.1, -.1, 1]
    np.random.seed(12)
    output = mutations.discrete_mutation(data)
    np.testing.assert_almost_equal(output, [-.33, -.1, .02], decimal=2)


def test_crossover() -> None:
    data = [0.1, -.1, 1, -0.1]
    np.random.seed(15)
    output = mutations.crossover(data, 2 * np.array(data))
    np.testing.assert_almost_equal(output, [-1.1, -.1, 2, -.1], decimal=2)


@testing.parametrized(
    dicrete=(mutations.discrete_mutation,),
    portfolio_discrete=(mutations.portfolio_discrete_mutation,),
    doubledoerr=(mutations.doubledoerr_discrete_mutation,),
    doerr=(mutations.doerr_discrete_mutation,),
)
def test_run_with_array(func: Callable[..., Any]) -> None:
    data = [0.1, -.1, 1, -.2] * 3
    np.random.seed(12)
    output1 = func(data)
    np.random.seed(12)
    output2 = func(np.array(data))
    np.testing.assert_equal(output1, output2)


@testing.parametrized(
    only_2=(2, 1.5),
    all_4=(4, 0.5),
)
def test_get_roulette(num: int, expected: str) -> None:
    np.random.seed(24)
    archive = utils.Archive[utils.Value]()
    for k in range(4):
        archive[np.array([k + .5])] = utils.Value(k)
    output = mutations.get_roulette(archive, num)
    np.testing.assert_equal(output, expected)
