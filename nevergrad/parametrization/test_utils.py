# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
    co = utils.Crossover(0, (0,))
    out = co.apply((x1, x2), rng=np.random.RandomState(12))
    expected = np.ones((2, 1)).dot([[5, 5, 4]])
    np.testing.assert_array_equal(out, expected)


def test_random_crossover() -> None:
    arrays = [k * np.ones((2, 2)) for k in range(31)]
    co = utils.Crossover(0)
    out = co.apply(arrays)
    assert 0 in out


@testing.parametrized(
    p2i2=(42, 2, 2, [0, 0, 1, 1, 1, 0]),
    p5i6=(42, 5, 6, [3, 0, 1, 2, 5, 4]),
    p1i2=(42, 1, 2, [0, 0, 1, 1, 1, 1]),
    p2i3=(42, 2, 3, [1, 1, 2, 2, 2, 0]),
    p3i2=(42, 2, 2, [0, 0, 1, 1, 1, 0]),
)
def test_kpoint_crossover(seed: int, points: int, indiv: int, expected: tp.List[int]) -> None:
    rng = np.random.RandomState(seed)
    crossover = utils.Crossover(points)
    donors = [k * np.ones(len(expected)) for k in range(indiv)]
    output = crossover.apply(donors, rng)
    np.testing.assert_array_equal(output, expected)


@testing.parametrized(
    small=(1, 5, [0]),
    keep_first=(2, 1000, [0, 871]),
    two_points=(3, 2, [0, 1, 0]),
    two_points_big=(3, 1000, [518, 871, 0]),
)
def test_make_crossover_sequence(num_sections: int, num_individuals: int, expected: tp.List[int]) -> None:
    rng = np.random.RandomState(12)
    out = utils._make_crossover_sequence(num_sections=num_sections, num_individuals=num_individuals, rng=rng)
    assert out == expected


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
