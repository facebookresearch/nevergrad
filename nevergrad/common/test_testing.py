# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
import platform
import subprocess
import typing as tp
from pathlib import Path
import pytest
import numpy as np
from . import testing


# decorators for tests which do not need to be run on Windows
skip_win = pytest.mark.skipif(sys.platform == "win32", reason="Internals")


@testing.parametrized(
    equal=([2, 3, 1], ""),
    missing=((1, 2), ["  - missing element(s): {3}."]),
    additional=((1, 4, 3, 2), ["  - additional element(s): {4}."]),
    both=((1, 2, 4), ["  - additional element(s): {4}.", "  - missing element(s): {3}."]),
)
def test_assert_set_equal(estimate: tp.Iterable[int], message: str) -> None:
    reference = {1, 2, 3}
    try:
        testing.assert_set_equal(estimate, reference)
    except AssertionError as error:
        if not message:
            raise AssertionError("An error has been raised while it should not.") from error
        np.testing.assert_equal(error.args[0].split("\n")[1:], message)
    else:
        if message:
            raise AssertionError("An error should have been raised.")


def test_printed_assert_equal() -> None:
    testing.printed_assert_equal(0, 0)
    np.testing.assert_raises(AssertionError, testing.printed_assert_equal, 0, 1)


@skip_win  # type: ignore
def test_assert_markdown_links_not_broken() -> None:
    folder = Path(__file__).parents[2].expanduser().absolute()
    assert (folder / "README.md").exists(), f"Wrong root folder: {folder}"
    assert testing._get_all_markdown_links(folder), "There should be at least one hyperlink!"
    testing.assert_markdown_links_not_broken(folder)


@testing.parametrized(
    changed=(RuntimeError, platform.system(), unittest.SkipTest),
    wrong_system=(RuntimeError, "blublu", RuntimeError),
    wrong_error=(AssertionError, platform.system(), RuntimeError),
)
def test_skip_test_on_system(
    skipped_error: tp.Type[Exception], system: str, expected_error: tp.Type[Exception]
) -> None:
    try:
        with pytest.raises(expected_error):
            with testing.skip_error_on_systems(skipped_error, (system,)):
                raise RuntimeError("Testing skip")
    except unittest.SkipTest as e:  # prevents SkipTest from just skipping the test and making it useless
        raise AssertionError("Should not have skipped the test!") from e


@skip_win  # type: ignore
def test_header() -> None:
    header = Path(__file__).read_text().splitlines()[0]
    repopath = Path(__file__).parents[1]
    assert repopath.name == "nevergrad"
    assert (repopath.parent / "setup.py").exists()
    output = subprocess.check_output(["find", str(repopath), "-name", "*.py"], shell=False).decode()
    missing: tp.List[str] = []
    for filepath in output.splitlines():
        if not Path(filepath).read_text().startswith(header):
            missing.append(filepath)
    if missing:
        missing_str = "\n - ".join(missing)
        raise AssertionError(
            f"Following files are missing standard header (see other files):\n - {missing_str}"
        )
