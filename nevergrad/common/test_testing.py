# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Iterable
import numpy as np
from . import testing


@testing.parametrized(
    equal=([2, 3, 1], ""),
    missing=((1, 2), ["  - missing element(s): {3}."]),
    additional=((1, 4, 3, 2), ["  - additional element(s): {4}."]),
    both=((1, 2, 4), ["  - additional element(s): {4}.", "  - missing element(s): {3}."]),
)
def test_assert_set_equal(estimate: Iterable[int], message: str) -> None:
    reference = {1, 2, 3}
    try:
        testing.assert_set_equal(estimate, reference)
    except AssertionError as error:
        if not message:
            raise AssertionError("An error has been raised while it should not.")
        np.testing.assert_equal(error.args[0].split("\n")[1:], message)
    else:
        if message:
            raise AssertionError("An error should have been raised.")


def test_printed_assert_equal() -> None:
    testing.printed_assert_equal(0, 0)
    np.testing.assert_raises(AssertionError, testing.printed_assert_equal, 0, 1)


def test_assert_markdown_links_not_broken() -> None:
    folder = Path(__file__).parents[2].expanduser().absolute()
    assert (folder / "README.md").exists(), f"Wrong root folder: {folder}"
    assert testing._get_all_markdown_links(folder), "There should be at least one hyperlink!"
    testing.assert_markdown_links_not_broken(folder)
