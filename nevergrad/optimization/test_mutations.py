# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from typing import Callable, Any
import numpy as np
import genty
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


@genty.genty
class MutationTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        dicrete=(mutations.discrete_mutation,),
        portfolio_discrete=(mutations.portfolio_discrete_mutation,),
        doubledoerr=(mutations.doubledoerr_discrete_mutation,),
        doerr=(mutations.doerr_discrete_mutation,),
    )
    def test_run_with_array(self, func: Callable[..., Any]) -> None:
        data = [0.1, -.1, 1, -.2] * 3
        np.random.seed(12)
        output1 = func(data)
        np.random.seed(12)
        output2 = func(np.array(data))
        np.testing.assert_equal(output1, output2)

    @genty.genty_dataset(  # type: ignore
        only_2=(2, 1.5),
        all_4=(4, 0.5),
    )
    def test_get_roulette(self, num: int, expected: str) -> None:
        np.random.seed(24)
        archive = utils.Archive()
        for k in range(4):
            archive[np.array([k + .5])] = utils.Value(k)
        output = mutations.get_roulette(archive, num)
        np.testing.assert_equal(output, expected)
