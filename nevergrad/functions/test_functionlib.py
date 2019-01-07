# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from typing import Any, Dict
import genty
import numpy as np
from ..common import testing
from . import functionlib


DESCRIPTION_KEYS = {"function_class", "name", "block_dimension", "useful_dimensions", "useless_variables", "translation_factor",
                    "num_blocks", "rotation", "noise_level", "dimension", "discrete", "aggregator", "hashing", "transform"}


def test_testcase_function_errors() -> None:
    config: Dict[str, Any] = {"name": "blublu", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2}
    np.testing.assert_raises(ValueError, functionlib.ArtificialFunction, **config)
    config["num_blocks"] = 0
    np.testing.assert_raises(AssertionError, functionlib.ArtificialFunction, **config)


def test_artitificial_function_repr() -> None:
    config: Dict[str, Any] = {"name": "sphere", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2}
    func = functionlib.ArtificialFunction(**config)
    output = repr(func)
    assert "sphere" in output, f"Unexpected representation: {output}"


def test_testcase_function_value() -> None:
    # make sure no change is made to the computation
    config: Dict[str, Any] = {"name": "sphere", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2}
    np.random.seed(2)
    x = np.random.normal(0, 1, 12)
    np.random.seed(12)
    func = functionlib.ArtificialFunction(**config)
    value = func(x)
    np.testing.assert_almost_equal(value, 9.630, decimal=3)


@genty.genty
class TestcaseTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        random=(np.random.normal(0, 1, 12), False),
        hashed=("abcdefghijkl", True),
    )
    def test_test_function(self, x: Any, hashing: bool) -> None:
        config: Dict[str, Any] = {"name": "sphere", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2, "hashing": hashing}
        outputs = []
        for _ in range(2):
            np.random.seed(12)
            func = functionlib.ArtificialFunction(**config)
            outputs.append(func(x))
        np.testing.assert_equal(outputs[0], outputs[1])
        # make sure it is properly random otherwise
        outputs.append(functionlib.ArtificialFunction(**config)(x))
        assert outputs[1] != outputs[2]


def test_oracle() -> None:
    func = functionlib.ArtificialFunction("sphere", 5, noise_level=.1)
    x = [1, 2, 1, 0, .5]
    y1 = func(x)  # returns a float
    y2 = func(x)  # returns a different float since the function is noisy
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, y1, y2)
    y3 = func.oracle_call(x)   # returns a float
    y4 = func.oracle_call(x)   # returns the same float (no noise for oracles + sphere function is deterministic)
    np.testing.assert_array_almost_equal(y3, y4)  # should be different
    func = functionlib.ArtificialFunction("sphere", 5, noise_level=.1)
    y5 = func.oracle_call(x)   # returns a different float than before, because a random translation is applied
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, y4, y5)


def test_artificial_function_summary() -> None:
    func = functionlib.ArtificialFunction("sphere", 5)
    testing.assert_set_equal(func.descriptors.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(func.descriptors["function_class"], "ArtificialFunction")


def test_duplicate() -> None:
    func = functionlib.ArtificialFunction("sphere", 5, noise_level=.2, num_blocks=4)
    func2 = func.duplicate()
    assert func == func2
    assert func._noise_level == func2._noise_level
    assert func is not func2


def test_artifificial_function_with_jump() -> None:
    func1 = functionlib.ArtificialFunction("sphere", 5)
    func2 = functionlib.ArtificialFunction("jump5", 5)
    np.testing.assert_equal(func1._only_index_transform, False)
    np.testing.assert_equal(func2._only_index_transform, True)
