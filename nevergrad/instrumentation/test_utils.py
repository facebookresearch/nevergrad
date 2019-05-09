# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import contextlib
from pathlib import Path
from typing import List, Any
import numpy as np
from ..common import testing
from . import variables
from . import utils


@testing.parametrized(
    empty=([], [], [])
)
def test_split_data(tokens: List[utils.Variable[Any]], data: List[float], expected: List[List[float]]) -> None:
    output = utils.split_data(data, tokens)
    testing.printed_assert_equal(output, expected)


def test_process_variables() -> None:
    tokens: List[utils.Variable[Any]] = [variables.SoftmaxCategorical(list(range(5))),  # TODO: why casting?
                                         variables.Gaussian(3, 4)]
    values = utils.process_variables(tokens, [0, 200, 0, 0, 0, 2])
    np.testing.assert_equal(values, [1, 11])
    np.testing.assert_raises(AssertionError, utils.process_variables, tokens, [0, 200, 0, 0, 0, 2, 3])


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
