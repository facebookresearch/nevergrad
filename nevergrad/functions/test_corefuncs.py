# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from typing import Callable
import numpy as np
import genty
from . import corefuncs


@genty.genty
class CoreFuncsTests(TestCase):

    @genty.genty_dataset(**{name: (name, func) for name, func in corefuncs.registry.items()})  # type: ignore
    def test_core_function(self, name: str, func: Callable) -> None:
        x = np.random.normal(0, 1, 100)
        outputs = []
        for _ in range(2):
            np.random.seed(12)
            outputs.append(func(x))
        np.testing.assert_equal(outputs[0], outputs[1], f'Function {name} is not deterministic')

    @genty.genty_dataset(  # type: ignore
        expe1=([.6, .4, .2, .1, .9], 3, 4, 2),  # jump was assumed correct (verify?)
        expe2=([.6, .6, .7, .1, .9], 1, 2, 0),
    )
    def test_base_functions(self, x: np.ndarray, onemax_expected: float, leadingones_expected: float, jump_expected: float) -> None:
        np.testing.assert_equal(corefuncs._onemax(x), onemax_expected, err_msg="Wrong output for onemax")
        np.testing.assert_equal(corefuncs._leadingones(x), leadingones_expected, err_msg="Wrong output for leadingones")
        np.testing.assert_equal(corefuncs._jump(x), jump_expected, err_msg="Wrong output for jump")


def test_genzcornerpeak_inf() -> None:
    y = [-np.inf, -np.inf]
    output = corefuncs.genzcornerpeak(y)
    np.testing.assert_equal(output, np.inf)
    output = corefuncs.minusgenzcornerpeak(y)
    np.testing.assert_equal(output, -np.inf)
