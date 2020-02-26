# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import contextlib
import typing as tp
from pathlib import Path
import numpy as np
from nevergrad.common import testing
from . import parameter as p
from . import utils
from . import helpers


def test_temporary_directory_copy() -> None:
    filepath = Path(__file__)
    with utils.TemporaryDirectoryCopy(filepath.parent) as cpath:
        assert cpath.exists()
        assert (cpath / filepath.name).exists()
    assert not cpath.exists()


def test_command_function() -> None:
    command = f"{sys.executable} -m nevergrad.parametrization.test_utils".split()
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


@testing.parametrized(
    scalar=(False, p.Scalar(), ("",)),
    v_scalar=(True, p.Scalar(), ("",)),
    tuple_=(False, p.Tuple(p.Scalar(), p.Array(shape=(2,))), ("", "0", "1")),
    v_tuple_=(True, p.Tuple(p.Scalar(), p.Array(shape=(2,))), ("0", "1")),
    instrumentation=(False, p.Instrumentation(p.Scalar(), y=p.Scalar()), ("", "0", "y")),
    instrumentation_v=(True, p.Instrumentation(p.Scalar(), y=p.Scalar()), ("0", "y")),
    choice=(False, p.Choice([p.Scalar(), "blublu"]), ("", "choices", "choices.0", "choices.1", "weights")),
    v_choice=(True, p.Choice([p.Scalar(), "blublu"]), ("", "choices.0", "weights")),
    tuple_choice_dict=(False, p.Tuple(p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar()])),
                       ("", "0", "0.choices", "0.choices.0", "0.choices.0.x", "0.choices.0.y", "0.choices.1", "0.weights")),
    v_tuple_choice_dict=(True, p.Tuple(p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar()])),
                         ("0", "0.choices.0.x", "0.choices.1", "0.weights")),
)
def test_flatten_parameter(no_container: bool, param: p.Parameter, keys: tp.Iterable[str]) -> None:
    flat = helpers.flatten_parameter(param, with_containers=not no_container)
    assert set(flat) == set(keys), f"Unexpected flattened parameter: {flat}"


@testing.parametrized(
    order_0=(0, ("", "choices.0.x", "choices.1", "weights")),
    order_1=(1, ("", "choices.0.x", "choices.1", "weights", "choices.1#sigma", "choices.0.x#sigma")),
    order_2=(2, ("", "choices.0.x", "choices.1", "weights", "choices.1#sigma", "choices.0.x#sigma", "choices.1#sigma#sigma")),
    order_3=(3, ("", "choices.0.x", "choices.1", "weights", "choices.1#sigma", "choices.0.x#sigma", "choices.1#sigma#sigma")),
)
def test_flatten_parameter_order(order: int, keys: tp.Iterable[str]) -> None:
    param = p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar().sigma.set_mutation(sigma=p.Scalar())])
    flat = helpers.flatten_parameter(param, with_containers=False, order=order)
    assert set(flat) == set(keys), f"Unexpected flattened parameter: {flat}"


def test_crossover() -> None:
    x1 = 4 * np.ones((2, 3))
    x2 = 5 * np.ones((2, 3))
    co = utils.Crossover(axis=1)
    out = co._apply_array((x1, x2), rng=np.random.RandomState(12))
    expected = np.ones((2, 1)).dot([[4, 5, 4]])
    np.testing.assert_array_equal(out, expected)


def test_rolling() -> None:
    x = np.arange(4)[:, None].dot(np.ones((1, 2)))
    roll = utils.Rolling(0)
    out = roll._apply_array([x], rng=np.random.RandomState(12))
    expected = np.array([1, 2, 3, 0])[:, None].dot(np.ones((1, 2)))
    np.testing.assert_array_equal(out, expected)
    assert repr(roll) == "Rolling(axis=(0,))"


@testing.parametrized(
    all_none=(None, None),
    d2=((1, 2), None),
    d1=((1), None),
)
def test_crossover_axis(axis: tp.Optional[tp.Tuple[int, ...]], max_size: tp.Optional[int]) -> None:
    shape = (6, 8, 10)
    x1 = 4 * np.ones(shape)
    x2 = 5 * np.ones(shape)
    co = utils.Crossover(axis=axis, max_size=max_size)
    out = co._apply_array((x1, x2), rng=np.random.RandomState(12))
    np.testing.assert_array_equal(out.shape, shape)  # this basically only test that it did not raise an error


def test_descriptors() -> None:
    desc = utils.Descriptors(ordered=False)
    assert repr(desc) == "Descriptors(ordered=False)"


def do_nothing(*args: tp.Any, **kwargs: tp.Any) -> int:
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
