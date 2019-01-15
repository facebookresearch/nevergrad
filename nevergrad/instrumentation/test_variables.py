# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from typing import Any
import numpy as np
import genty
from ..common import testing
from .utils import vartypes
from . import variables


@genty.genty
class TokenTests(TestCase):

    @genty.genty_dataset(**{name: (name, token) for name, token in vartypes.items()})  # type: ignore
    def test_tokens(self, name: str, token_cls: Any) -> None:
        token = token_cls.from_str(token_cls.example)
        token.process(np.random.randn(token.dimension))
        token2 = token_cls.from_str(token_cls.example)
        assert token == token2
        assert repr(token).startswith(name)
        np.testing.assert_equal(repr(token), repr(token2))
        np.testing.assert_equal(token.example[:3], "NG_", err_msg="InstrumentizedFile assumption is broken")


def test_soft_discrete() -> None:
    np.random.seed(12)
    token = variables.SoftmaxCategorical(["blu", "blublu", "blublublu"])
    np.testing.assert_equal(token.process([.5, 1, 2.]), "blublu")
    np.testing.assert_equal(token.process(token.process_arg("blu"), deterministic=True), "blu")


def test_hard_discrete() -> None:
    token = variables.OrderedDiscrete(["blu", "blublu", "blublublu"])
    np.testing.assert_equal(token.process([5]), "blublublu")
    np.testing.assert_equal(token.process([0]), "blublu")
    np.testing.assert_equal(token.process(token.process_arg("blu"), deterministic=True), "blu")


def test_gaussian() -> None:
    token = variables.Gaussian(1, 3)
    np.testing.assert_equal(token.process([.5]), 2.5)
    np.testing.assert_equal(token.process(token.process_arg(12)), 12)


def test_instrumentation() -> None:
    instru = variables.Instrumentation(variables.Gaussian(0, 1),
                                       3,
                                       b=variables.SoftmaxCategorical([0, 1, 2, 3]),
                                       a=variables.OrderedDiscrete([0, 1, 2, 3]))
    np.testing.assert_equal(instru.dimension, 6)
    data = instru.arguments_to_data(4, 3, a=0, b=3)
    np.testing.assert_array_almost_equal(data, [4, -1.1503, 0, 0, 0, .5878], decimal=4)
    args, kwargs = instru.data_to_arguments(data, deterministic=True)
    testing.printed_assert_equal((args, kwargs), ((4., 3), {'a': 0, 'b': 3}))
    # check deterministic
    data = [0, 0, 0, 0, 0, 0]
    total = 0
    for _ in range(24):
        total += instru.data_to_arguments(data, deterministic=True)[1]["b"]
    np.testing.assert_equal(total, 0)
    # check stochastic
    for _ in range(24):
        total += instru.data_to_arguments(data, deterministic=False)[1]["b"]
    assert total != 0
