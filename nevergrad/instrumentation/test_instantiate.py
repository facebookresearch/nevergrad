# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from ..common import testing
from . import instantiate
from .instantiate import LINETOKEN
from .instantiate import Placeholder


def test_symlink_folder_tree() -> None:
    path = Path(__file__).absolute().parents[1]
    assert path.name == "nevergrad"
    with tempfile.TemporaryDirectory() as folder:
        instantiate.symlink_folder_tree(path, folder)


_EXPECTED = [Placeholder(*x) for x in [("value1", "this is a comment"), ("value2", None), ("string", None)]]


def _check_uncomment_line(line: str, ext: str, expected: str) -> None:
    if isinstance(expected, str):
        output = instantiate.uncomment_line(line, ext)
        np.testing.assert_equal(output, expected)
    else:
        np.testing.assert_raises(expected, instantiate.uncomment_line, line, ext)


# CAREFUL: avoid triggering errors if the module parses itself...
# Note: 'bidule' is French for dummy widget
@testing.parametrized(
    nothing=("    bidule", ".py", "    bidule"),
    python=(f"    # {LINETOKEN} bidule", ".py", "    bidule"),
    not_starting_python=(f"x    # {LINETOKEN} bidule", ".py", RuntimeError),
    bad_python=("    // @" + "nevergrad@ bidule", ".py", RuntimeError),
    cpp=(f"  //{LINETOKEN}bidule", ".cpp", "  bidule"),
    matlab=(f"%{LINETOKEN}bidule", ".m", "bidule"),
    unknown=(f"// {LINETOKEN} bidule", ".unknown", RuntimeError),
)
def test_uncomment_line(line: str, ext: str, expected: str) -> None:
    _check_uncomment_line(line, ext, expected)


@testing.parametrized(
    custom=(f"// {LINETOKEN} bidule", ".custom", "//", "bidule"),
    wrong_comment_chars=(f"// {LINETOKEN} bidule", ".custom", "#", RuntimeError),
)
def test_uncomment_line_custom_file_type(line: str, ext: str, comment: str, expected: str) -> None:
    instantiate.FolderFunction.register_file_type(ext, comment)
    _check_uncomment_line(line, ext, expected)
    del instantiate.COMMENT_CHARS[ext]


@testing.parametrized(
    with_clean_copy=(True,),
    without_clean_copy=(False,),
)
def test_folder_instantiator(clean_copy: bool) -> None:
    path = Path(__file__).parent / "examples"
    ifolder = instantiate.FolderInstantiator(path, clean_copy=clean_copy)
    testing.printed_assert_equal(ifolder.placeholders, _EXPECTED)
    np.testing.assert_equal(len(ifolder.file_functions), 1)
    with ifolder.instantiate(value1=12, value2=110., string="") as tmp:
        with (tmp / "script.py").open("r") as f:
            lines = f.readlines()
    np.testing.assert_equal(lines[10], "value2 = 110.0\n")


@testing.parametrized(
    void=("bvcebsl\nsoefn", []),
    unique_no_comment=("bfseibf\nbsfei NG_ARG{machin}", [("machin", None)]),
    several=("bfkes\nsgrdgrgbdrkNG_ARG{truc|blublu}sehnNG_ARG{bidule}", [("truc", "blublu"), ("bidule", None)]),
)
def test_placeholder(text: str, name_comments: List[Tuple[str, Optional[str]]]) -> None:
    placeholders = Placeholder.finditer(text)
    testing.printed_assert_equal(placeholders, [Placeholder(*x) for x in name_comments])


@testing.parametrized(
    python=(".py", "[[1, 2], [3, 4]]"),
    cpp=(".cpp", "{{1, 2}, {3, 4}}"),
)
def test_placeholder_for_array(extension: str, expected: str) -> None:
    text = "NG_ARG{bidule}"
    output = Placeholder.sub(text, extension, {"bidule": np.array([[1, 2], [3, 4]])})
    np.testing.assert_equal(output, expected)


def test_placeholder_substitution() -> None:
    text = "bfkes\nsgrdgrgbdrkNG_ARG{truc|blublu}sehn NG_ARG{bidule}"
    expected = "bfkes\nsgrdgrgbdrk'#12#'sehn 24"
    output = Placeholder.sub(text, ".py", {"truc": "#12#", "bidule": 24})
    np.testing.assert_equal(output, expected)
    np.testing.assert_raises(KeyError, Placeholder.sub, text, ".py", {"truc": "#12#"})
    np.testing.assert_raises(RuntimeError, Placeholder.sub, text, ".py", {"truc": "#12#", "bidule": 24, "chouette": 2})
    text = "bfkes\nsgrdgrgbdrkNG_ARG{truc|blublu}sehnNG_ARG{bidule}NG_ARG{bidule|bis}"
    np.testing.assert_raises(RuntimeError, Placeholder.sub, text, ".py", {"truc": "#12#", "bidule": 24})


def test_file_text_function() -> None:
    path = Path(__file__).parent / "examples" / "script.py"
    filefunc = instantiate.FileTextFunction(path)
    testing.printed_assert_equal(filefunc.placeholders, _EXPECTED)


def test_folder_function() -> None:
    folder = Path(__file__).parent / "examples"
    func = instantiate.FolderFunction(str(folder), ["python", "examples/script.py"], clean_copy=True)
    output = func(value1=98, value2=12, string="plop")
    np.testing.assert_equal(output, 24)
    output = func(value1=98, value2=12, string="blublu")
    np.testing.assert_equal(output, 12)
