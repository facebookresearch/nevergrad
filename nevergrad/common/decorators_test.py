# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
import numpy as np
from . import decorators


class DecoratorTests(TestCase):

    def test_registry(self) -> None:
        functions = decorators.Registry()
        REGISTERED = functions._registered  # for compatibility... to be removed
        other = decorators.Registry()

        @functions.register
        def dummy() -> int:
            return 12

        np.testing.assert_equal(dummy(), 12)
        np.testing.assert_array_equal(list(functions.keys()), ["dummy"])
        np.testing.assert_array_equal([x.__name__ for x in REGISTERED], ["dummy"])
        np.testing.assert_array_equal(list(other.keys()), [])

    def test_info_registry(self) -> None:
        functions = decorators.Registry()

        @functions.register_with_info(tag="info")
        def dummy_info() -> int:
            return 10

        np.testing.assert_equal(dummy_info(), 10)
        np.testing.assert_equal(functions.get_info("dummy_info"), {"tag": "info"})
        np.testing.assert_raises(ValueError, functions.get_info, "no_dummy")

    def test_registry_error(self) -> None:
        functions = decorators.Registry()

        @functions.register
        def dummy() -> int:
            return 12

        np.testing.assert_raises(RuntimeError, functions.register, dummy)
