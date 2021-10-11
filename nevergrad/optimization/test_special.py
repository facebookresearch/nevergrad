# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import typing as tp
import pytest
from . import test_optimizerlib


KEY = "NEVERGRAD_SPECIAL_TESTS"
if not os.environ.get(KEY, ""):
    pytest.skip(f"These tests only run if {KEY} is set in the environment", allow_module_level=True)


@pytest.mark.parametrize("args", test_optimizerlib.get_metamodel_test_settings(special=True))
@pytest.mark.parametrize("baseline", ("CMA", "ECMA"))
def test_metamodel_special(baseline: str, args: tp.Tuple[tp.Any, ...]) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    kwargs = dict(zip(test_optimizerlib.META_TEST_ARGS, args))
    test_optimizerlib.check_metamodel(baseline=baseline, **kwargs)
