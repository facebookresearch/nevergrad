# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable, Any
import numpy as np


def assert_set_equal(estimate: Iterable, reference: Iterable, err_msg: str = "") -> None:
    """Asserts that both sets are equals, with comprehensive error message.
    This function should only be used in tests.
    Parameters
    ----------
    estimate: iterable
        sequence of elements to compare with the reference set of elements
    reference: iterable
        reference sequence of elements
    """
    estimate, reference = (set(x) for x in [estimate, reference])
    elements = [("additional", estimate - reference), ("missing", reference - estimate)]
    messages = ["  - {} element(s): {}.".format(name, s) for (name, s) in elements if s]
    if messages:
        messages = ([err_msg] if err_msg else []) + ["Sets are not equal:"] + messages
        raise AssertionError("\n".join(messages))


def printed_assert_equal(actual: Any, desired: Any, err_msg: str = '') -> None:
    try:
        np.testing.assert_equal(actual, desired, err_msg=err_msg)
    except AssertionError as e:
        print("\n" + "# " * 12 + "DEBUG MESSAGE " + "# " * 12)
        print(f"Expected: {desired}\nbut got:  {actual}")
        raise e
