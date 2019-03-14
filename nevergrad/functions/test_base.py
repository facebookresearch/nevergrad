# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .base import ArtificiallyNoisyBaseFunction
from ..common import testing


class DummyFunction(ArtificiallyNoisyBaseFunction):

    _TRANSFORMS = {"tanh": lambda s, x: np.tanh(x)}  # type:ignore

    def oracle_call(self, x: np.ndarray) -> float:
        return float(np.arctanh(x)[0])


@testing.parametrized(
    no_noise=(2, False, False, False),
    noise=(2, True, False, True),
    noise_dissymmetry_pos=(2, True, True, False),  # no noise on right side
    noise_dissymmetry_neg=(-2, True, True, True),
    no_noise_with_dissymmetry_neg=(-2, False, True, False),
)
def test_noise_addition(x: int, noise: bool, noise_dissymmetry: bool, expect_noisy: bool) -> None:
    func = DummyFunction(dimension=1, transform="tanh", noise_level=int(noise), noise_dissymmetry=noise_dissymmetry)
    fx = func([x])
    assert not np.isnan(fx)  # noise addition should not get out of function domain
    if expect_noisy:
        np.testing.assert_raises(AssertionError, np.testing.assert_almost_equal, fx, x, decimal=8)
    else:
        np.testing.assert_almost_equal(fx, x, decimal=8)
