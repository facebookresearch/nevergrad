# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from pathlib import Path
from typing import Iterable, Any, Union, List
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


def assert_markdown_links_not_broken(folder: Union[str, Path]) -> None:
    """Asserts that all relative hyperlinks are valid in markdown files of the folder
    and its subfolders.

    Note
    ----
    http hyperlinks are not tested.
    """
    links = _get_all_markdown_links(folder)
    broken = [l for l in links if not l.exists()]
    if broken:
        text = "\n - ".join([str(l) for l in broken])
        raise AssertionError(f"Broken markdown links:\n - {text}")


class _MarkdownLink:
    """Handle to a markdown link, for easy existence test and printing
    (external links are not tested)
    """

    def __init__(self, folder: Path, filepath: Path, string: str, link: str) -> None:
        self._folder = folder
        self._filepath = filepath
        self._string = string
        self._link = link

    def exists(self) -> bool:
        if self._link.startswith("http"):  # consider it exists
            return True
        fullpath = self._folder / self._filepath.parent / self._link
        return fullpath.exists()

    def __repr__(self) -> str:
        return f"{self._link} ({self._string}) from file {self._filepath}"


def _get_all_markdown_links(folder: Union[str, Path]) -> List[_MarkdownLink]:
    """Returns a list of all existing markdown links
    """
    pattern = re.compile(r"\[(?P<string>.+?)\]\((?P<link>.+?)\)")
    folder = Path(folder).expanduser().absolute()
    links = []
    for rfilepath in folder.glob("**/*.md"):
        filepath = folder / rfilepath
        with filepath.open("r") as f:
            text = f.read()
        for match in pattern.finditer(text):
            links.append(_MarkdownLink(folder, rfilepath, match.group("string"), match.group("link")))
    return links
