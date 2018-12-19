# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import contextlib
from unittest import TestCase
from pathlib import Path
from typing import List, Tuple, Any
import numpy as np
import genty
from ..common import testing
from . import variables
from . import utils


@genty.genty
class UtilsTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        all_fine=(2, ["a", "b", "c"], "blublu a c\nb"),
        too_many_values=(2, ["a", "b", "c", "d"], AssertionError),
        too_few_values=(2, ["a", "b"], IndexError),
        repeating_index=(1, ["a", "b", "c"], RuntimeError),
    )
    def test_replace_placeholders_by_values(self, last_index: int, values: Tuple[str], expected: Any) -> None:
        text = "blublu <[placeholder_{}>] <[placeholder_{}>]\n<[placeholder_{}>]".format(0, last_index, 1)
        if isinstance(expected, str):
            output = utils.replace_placeholders_by_values(text, values)
            np.testing.assert_equal(output, expected)
        else:
            np.testing.assert_raises(expected, utils.replace_placeholders_by_values, text, values)

    @genty.genty_dataset(  # type: ignore
        empty=([], [], [])
    )
    def test_split_data(self, tokens: List, data: List, expected: List) -> None:
        output = utils.split_data(data, tokens)
        testing.printed_assert_equal(output, expected)


def test_process_instruments() -> None:
    tokens = [variables.SoftmaxCategorical(list(range(5))),
              variables.Gaussian(3, 4)]
    values = utils.process_instruments(tokens, [0, 200, 0, 0, 0, 2])
    np.testing.assert_equal(values, [1, 11])
    np.testing.assert_raises(AssertionError, utils.process_instruments, tokens, [0, 200, 0, 0, 0, 2, 3])


def test_replace_tokens_by_placeholders() -> None:
    intext = "blublu NG_SC{0|1|2} NG_G{2,3}\nNG_SC{a|b}"
    outtext, tokens = utils.replace_tokens_by_placeholders(intext)
    expected_text = "blublu <[placeholder_{}>] <[placeholder_{}>]\n<[placeholder_{}>]".format(1, 0, 2)
    expected_vars = [variables.Gaussian(mean=2.0, std=3),
                     variables.SoftmaxCategorical(possibilities=["0", "1", "2"]),
                     variables.SoftmaxCategorical(possibilities=["a", "b"])]
    np.testing.assert_equal(outtext, expected_text)
    np.testing.assert_array_equal(tokens, expected_vars)


def test_temporary_directory_copy() -> None:
    filepath = Path(__file__)
    with utils.TemporaryDirectoryCopy(filepath.parent) as cpath:
        assert cpath.exists()
        assert (cpath / filepath.name).exists()
    assert not cpath.exists()


def test_command_function() -> None:
    command = "python -m nevergrad.instrumentation.test_utils".split()
    word = "testblublu12"
    output = utils.CommandFunction(command)(word)
    assert output is not None
    assert word in output, f'Missing word "{word}" in output:\n{output}'
    try:
        with contextlib.redirect_stderr(sys.stdout):
            output = utils.CommandFunction(command, verbose=True)(error=True)
    except utils.FailedJobError as e:
        words = "Too bad"
        assert words in str(e), f'Missing word "{words}" in output:\n\n{e}'
    else:
        raise AssertionError("An error should have been raised")


def do_nothing(*args: Any, **kwargs: Any) -> int:
    print("my args", args, flush=True)
    print("my kwargs", kwargs, flush=True)
    if "sleep" in kwargs:
        print("Waiting", flush=True)
        time.sleep(int(kwargs["sleep"]))
    if kwargs.get("error", False):
        print("Raising", flush=True)
        raise ValueError("Too bad")
    print("Finishing", flush=True)
    return 12


if __name__ == "__main__":
    c_args, c_kwargs = [], {}  # oversimplisitic parser
    for argv in sys.argv[1:]:
        if "=" in argv:
            key, val = argv.split("=")
            c_kwargs[key.strip("-")] = val
        else:
            c_args.append(argv)
    do_nothing(*c_args, **c_kwargs)
