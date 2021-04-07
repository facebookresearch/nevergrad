# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import os
import sys
import time
import random
import itertools
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
    # if os.environ.get("CIRCLECI", False):
    #     raise SkipTest("Failing in CircleCI")  # TODO investigate why
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
    choice=(False, p.Choice([p.Scalar(), "blublu"]), ("", "choices", "choices.0", "choices.1", "indices")),
    v_choice=(True, p.Choice([p.Scalar(), "blublu"]), ("", "choices.0", "indices")),
    tuple_choice_dict=(
        False,
        p.Tuple(p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar()])),
        ("", "0", "0.choices", "0.choices.0", "0.choices.0.x", "0.choices.0.y", "0.choices.1", "0.indices"),
    ),
    v_tuple_choice_dict=(
        True,
        p.Tuple(p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar()])),
        ("0", "0.choices.0.x", "0.choices.1", "0.indices"),
    ),
)
def test_flatten(no_container: bool, param: p.Parameter, keys: tp.Iterable[str]) -> None:
    flat = dict(helpers.flatten(param, with_containers=not no_container))
    assert set(flat) == set(keys), f"Unexpected flattened parameter: {flat}"


def test_function_info() -> None:
    info = utils.FunctionInfo(deterministic=False)
    assert repr(info) == "FunctionInfo(deterministic=False,metrizable=True,proxy=False)"


@testing.parametrized(
    # updating this tests requires checking manually through prints
    # that everything works as intended
    v_tuple_choice_dict=(
        p.Tuple(p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar()])),
        ["0.choices.0.x", "0.choices.1", "0.indices"],
    ),
    multiple=(
        p.Instrumentation(
            p.Scalar(init=12, lower=12, upper=12.01),
            x=p.Choice([3, p.Log(lower=0.01, upper=0.1)]),
            z=p.Array(init=[12, 12]).set_bounds(lower=12, upper=15),
            y=p.Array(init=[1, 1]),
        ),
        ["0", "x.choices.1", "x.indices", "y", "z"],
    ),
)
def test_split_as_data_parameters(param: p.Parameter, names: tp.List[str]) -> None:
    # new version
    output = helpers.flatten(param)
    assert [x[0] for x in output if isinstance(x[1], p.Data)] == names
    # legacy
    output2 = split_as_data_parameters(param)
    assert [x[0] for x in output2] == names


@testing.parametrized(
    order_0=(0, ("", "choices.0.x", "choices.1", "indices")),
    order_1=(1, ("", "choices.0.x", "choices.1", "indices", "choices.1#sigma", "choices.0.x#sigma")),
    order_2=(
        2,
        (
            "",
            "choices.0.x",
            "choices.1",
            "indices",
            "choices.1#sigma",
            "choices.0.x#sigma",
            "choices.1#sigma#sigma",
        ),
    ),
    order_3=(
        3,
        (
            "",
            "choices.0.x",
            "choices.1",
            "indices",
            "choices.1#sigma",
            "choices.0.x#sigma",
            "choices.1#sigma#sigma",
        ),
    ),
)
def test_flatten_order(order: int, keys: tp.Iterable[str]) -> None:
    param = p.Choice([p.Dict(x=p.Scalar(), y=12), p.Scalar().sigma.set_mutation(sigma=p.Scalar())])
    flat = dict(helpers.flatten(param, with_containers=False, order=order))
    assert set(flat) == set(keys), f"Unexpected flattened parameter: {flat}"


@testing.parametrized(
    true=(True, 0.0),
    false=(False, 1.0),
    np_true=(np.bool_(True), 0.0),
    np_false=(np.bool_(False), 1.0),
    pos=(0.7, 0.0),
    neg=(-0.7, 0.7),
    np_pos=(np.float(0.7), 0.0),
    np_neg=(np.float(-0.7), 0.7),
)
def test_float_penalty(value: tp.Any, expected: float) -> None:
    assert utils.float_penalty(value) == expected


# # # OLD FUNCTION SERVING AS CHECK FOR DATA ORDER # # #


# pylint: disable=too-many-locals
def split_as_data_parameters(
    parameter: p.Parameter,
) -> tp.List[tp.Tuple[str, p.Data]]:
    """List all the instances involved as parameter (not as subparameter/
    endogeneous parameter) ordered as in standardized data space

    Parameter
    ---------
    parameter: Parameter
        the parameter to split

    Returns
    -------
    list
        the list and subparameters ordered as in data space

    Note
    ----
    This function is experimental, its output will probably evolve before converging.
    """
    err_msg = (
        f"Could not figure out the data order for: {parameter} "
        "(please open an issue on nevergrad github repository)"
    )
    copied = parameter.copy()
    ref = parameter.copy()
    flatp, flatc, flatref = (
        {x: y for x, y in helpers.flatten(pa) if isinstance(y, p.Data)} for pa in (parameter, copied, ref)
    )
    keys = list(flatp.keys())
    random.shuffle(keys)  # makes it safer to test!
    # remove transforms for both ref and copied parameters and set index
    for k, key in enumerate(keys):
        for not_ref, flat in enumerate((flatref, flatc)):
            param = flat[key]
            param._layers = param._layers[:1]  # force remove the bound to avoid weird clipping etc
            param.set_mutation(sigma=1.0)  # force mutation sigma to 1 to avoid rounding
            if not_ref:
                param.set_standardized_data(k * np.ones((param.dimension)))
    # analyze results
    data = copied.get_standardized_data(reference=ref)
    order: tp.List[int] = []
    for val, _ in itertools.groupby(data):
        num = int(np.round(val))
        if num in order:
            if order[-1] != num:
                raise RuntimeError(err_msg)
        else:
            order.append(num)
    if len(order) != len(flatp):
        raise RuntimeError(err_msg)
    # create output and check it
    ordered_keys = [keys[num] for num in order]
    ordered_arrays = [(k, flatp[k]) for k in ordered_keys]
    # print(f"DEBUGGING:\nkeys={keys}\ndata={data}\norder={order}\nordered_key={ordered_keys}")
    if sum(pa.dimension for _, pa in ordered_arrays) != parameter.dimension:
        raise RuntimeError(err_msg)
    return ordered_arrays


# # # END OF CHECK # # #


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
            key_, val_ = argv.split("=")
            c_kwargs[key_.strip("-")] = val_
        else:
            c_args.append(argv)
    do_nothing(*c_args, **c_kwargs)
