# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path
from unittest import TestCase
from typing import Tuple, Dict, Any
import genty
import numpy as np
from ..common import testing
from . import instantiate
from .instantiate import LINETOKEN
from . import variables
from . import utils


def test_symlink_folder_tree() -> None:
    path = Path(__file__).parents[2]
    with tempfile.TemporaryDirectory() as folder:
        instantiate.symlink_folder_tree(path, folder)


@genty.genty
class InstantiationTests(TestCase):

    def _test_uncomment_line(self, line: str, ext: str, expected: str) -> None:
        if isinstance(expected, str):
            output = instantiate.uncomment_line(line, ext)
            np.testing.assert_equal(output, expected)
        else:
            np.testing.assert_raises(expected, instantiate.uncomment_line, line, ext)

    # CAREFUL: avoid triggering errors if the module parses itself...
    # Note: 'bidule' is French for dummy widget
    @genty.genty_dataset(  # type: ignore
        nothing=("    bidule", ".py", "    bidule"),
        python=(f"    # {LINETOKEN} bidule", ".py", "    bidule"),
        not_starting_python=(f"x    # {LINETOKEN} bidule", ".py", RuntimeError),
        bad_python=("    // @" + "nevergrad@ bidule", ".py", RuntimeError),
        cpp=(f"  //{LINETOKEN}bidule", ".cpp", "  bidule"),
        matlab=(f"%{LINETOKEN}bidule", ".m", "bidule"),
        unknown=(f"// {LINETOKEN} bidule", ".unknown", RuntimeError),
    )
    def test_uncomment_line(self, line: str, ext: str, expected: str) -> None:
        self._test_uncomment_line(line, ext, expected)

    @genty.genty_dataset(  # type: ignore
        custom=(f"// {LINETOKEN} bidule", ".custom", "//", "bidule"),
        wrong_comment_chars=(f"// {LINETOKEN} bidule", ".custom", "#", RuntimeError),
    )
    def test_uncomment_line_custom_file_type(self, line: str, ext: str, comment: str, expected: str) -> None:
        instantiate.register_file_type(ext, comment)
        self._test_uncomment_line(line, ext, expected)
        del instantiate.COMMENT_CHARS[ext]

    @genty.genty_dataset(  # type: ignore
        with_clean_copy=(True,),
        without_clean_copy=(False,),
    )
    def test_folder_instantiator(self, clean_copy: bool) -> None:
        path = Path(__file__).parent / "examples" / "basic"
        ifolder = instantiate.FolderInstantiator(path, clean_copy=clean_copy)
        testing.printed_assert_equal(ifolder.placeholders, [utils.Placeholder("value1", "this is a comment"),
                                                            utils.Placeholder("value2", None)])
        np.testing.assert_equal(len(ifolder.file_functions), 1)
        with ifolder.instantiate(value1=12, value2=110.) as tmp:
            with (tmp / "script.py").open("r") as f:
                lines = f.readlines()
        np.testing.assert_equal(lines[10], "value2 = 110.0\n")


def test_file_text_function() -> None:
    path = Path(__file__).parent / "examples" / "basic" / "script.py"
    filefunc = instantiate.FileTextFunction(path)
    testing.printed_assert_equal(filefunc.placeholders, [utils.Placeholder("value1", "this is a comment"),
                                                         utils.Placeholder("value2", None)])


def _arg_return(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return args, kwargs


def test_instrumented_function() -> None:
    ifunc = instantiate.InstrumentedFunction(_arg_return, variables.SoftmaxCategorical([1, 12]), "constant",
                                             variables.Gaussian(0, 1, [2, 2]), constkwarg="blublu",
                                             plop=variables.SoftmaxCategorical([3, 4]))
    np.testing.assert_equal(ifunc.dimension, 8)
    data = [-100, 100, 1, 2, 3, 4, 100, -100]
    args, kwargs = ifunc(data)
    testing.printed_assert_equal(args, [12, "constant", [[1, 2], [3, 4]]])
    testing.printed_assert_equal(kwargs, {"constkwarg": "blublu", "plop": 3})
    testing.printed_assert_equal(ifunc.descriptors, {"dimension": 8, "name": "_arg_return", "instrumented": "arg0,arg2,plop",
                                                     "function_class": "InstrumentedFunction", "transform": None})
    print(ifunc.get_summary(data))


def test_instrumented_function_kwarg_order() -> None:
    ifunc = instantiate.InstrumentedFunction(_arg_return, kw4=variables.SoftmaxCategorical([1, 0]), kw2="constant",
                                             kw3=variables.Gaussian(0, 1, [2, 2]), kw1=variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.dimension, 7)
    data = [-1, 1, 2, 3, 4, 100, -100]
    _, kwargs = ifunc(data)
    testing.printed_assert_equal(kwargs, {"kw1": 0, "kw2": "constant", "kw3": [[1, 2], [3, 4]], "kw4": 1})


class _Callable:

    def __call__(self, x: float, y: float = 0) -> float:
        return abs(x + y)


def test_callable_instrumentation() -> None:
    ifunc = instantiate.InstrumentedFunction(lambda x: x**2, variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "<lambda>")
    ifunc = instantiate.InstrumentedFunction(_Callable(), variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "_Callable")


def test_deterministic_convert_to_args() -> None:
    ifunc = instantiate.InstrumentedFunction(_Callable(), variables.SoftmaxCategorical([0, 1, 2, 3]),
                                             y=variables.SoftmaxCategorical([0, 1, 2, 3]))
    data = [.01, 0, 0, 0, .01, 0, 0, 0]
    for _ in range(20):
        args, kwargs = ifunc.convert_to_arguments(data, deterministic=True)
        testing.printed_assert_equal(args, [0])
        testing.printed_assert_equal(kwargs, {"y": 0})
    arg_sum, kwarg_sum = 0, 0
    for _ in range(24):
        args, kwargs = ifunc.convert_to_arguments(data, deterministic=False)
        arg_sum += args[0]
        kwarg_sum += kwargs["y"]
    assert arg_sum != 0
    assert kwarg_sum != 0
