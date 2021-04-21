# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
import pytest
from nevergrad.common import testing
from nevergrad.parametrization import parameter as p
from . import functionlib


DESCRIPTION_KEYS = {
    "split",
    "function_class",
    "name",
    "block_dimension",
    "useful_dimensions",
    "useless_variables",
    "translation_factor",
    "num_blocks",
    "rotation",
    "noise_level",
    "dimension",
    "discrete",
    "aggregator",
    "hashing",
    "parametrization",
    "noise_dissymmetry",
}


def test_testcase_function_errors() -> None:
    config: tp.Dict[str, tp.Any] = {
        "name": "blublu",
        "block_dimension": 3,
        "useless_variables": 6,
        "num_blocks": 2,
    }
    np.testing.assert_raises(ValueError, functionlib.ArtificialFunction, **config)  # blublu does not exist
    config.update(name="sphere")
    functionlib.ArtificialFunction(**config)  # should wor
    config.update(num_blocks=0)
    np.testing.assert_raises(ValueError, functionlib.ArtificialFunction, **config)  # num blocks should be > 0
    config.update(num_blocks=2.0)
    np.testing.assert_raises(TypeError, functionlib.ArtificialFunction, **config)  # num blocks should be > 0
    config.update(num_blocks=2, rotation=1)
    np.testing.assert_raises(TypeError, functionlib.ArtificialFunction, **config)  # num blocks should be > 0


def test_artitificial_function_repr() -> None:
    config: tp.Dict[str, tp.Any] = {
        "name": "sphere",
        "block_dimension": 3,
        "useless_variables": 6,
        "num_blocks": 2,
    }
    func = functionlib.ArtificialFunction(**config)
    output = repr(func)
    assert "sphere" in output, f"Unexpected representation: {output}"


def test_ptb_no_overfitting() -> None:
    func = functionlib.PBT(("sphere", "cigar"), (3, 7), 12)
    func = func.copy()
    # We do a gradient descent.
    value = [func(-15.0 * np.ones(2)) for _ in range(1500)]
    # We check that the values are becoming better and better.
    assert value[-1] < value[len(value) // 2]  # type: ignore
    assert value[0] > value[len(value) // 2]  # type: ignore


@testing.parametrized(
    sphere=(
        {"name": "sphere", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2},
        13.377591870607294,
    ),
    cigar=(
        {"name": "cigar", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2},
        12492378.626191331,
    ),
    cigar_rot=(
        {"rotation": True, "name": "cigar", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2},
        2575881.272645816,
    ),
    hashed=(
        {"name": "sphere", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2, "hashing": True},
        8.916424986561422,
    ),
    noisy_sphere=(
        {"name": "sphere", "block_dimension": 3, "useless_variables": 6, "num_blocks": 2, "noise_level": 0.2},
        14.512132049518083,
    ),
    noisy_very_sphere=(
        {
            "name": "sphere",
            "block_dimension": 3,
            "useless_variables": 6,
            "num_blocks": 2,
            "noise_dissymmetry": True,
            "noise_level": 0.2,
        },
        19.33566196119778,
    ),
)
def test_testcase_function_value(config: tp.Dict[str, tp.Any], expected: float) -> None:
    # make sure no change is made to the computation
    func = functionlib.ArtificialFunction(**config)
    np.random.seed(1)  # don't know how to control to randomness
    func = func.copy()
    np.random.seed(2)  # initialization is delayed
    x = np.random.normal(0, 1, func.dimension)
    x *= -1 if config.get("noise_dissymmetry", False) else 1  # change sign to activate noise dissymetry
    np.random.seed(12)  # function randomness comes at first call
    value = func(x)
    np.testing.assert_almost_equal(value, expected, decimal=3)


@testing.parametrized(
    random=(np.random.normal(0, 1, 12), False),
    hashed=(np.ones(12), True),
)
def test_test_function(x: tp.Any, hashing: bool) -> None:
    config: tp.Dict[str, tp.Any] = {
        "name": "sphere",
        "block_dimension": 3,
        "useless_variables": 6,
        "num_blocks": 2,
        "hashing": hashing,
    }
    outputs = []
    for _ in range(2):
        np.random.seed(12)
        func = functionlib.ArtificialFunction(**config)
        outputs.append(func(x))
    np.testing.assert_equal(outputs[0], outputs[1])
    # make sure it is properly random otherwise
    outputs.append(functionlib.ArtificialFunction(**config)(x))
    assert outputs[1] != outputs[2]


def test_oracle() -> None:
    func = functionlib.ArtificialFunction("sphere", 5, noise_level=0.1)
    x = np.array([1, 2, 1, 0, 0.5])
    y1 = func(x)  # returns a float
    y2 = func(x)  # returns a different float since the function is noisy
    np.testing.assert_raises(AssertionError, np.testing.assert_array_almost_equal, y1, y2)
    reco = p.Array(init=x)
    y3 = func.evaluation_function(reco)  # returns a float
    # returns the same float (no noise for oracles + sphere function is deterministic)
    y4 = func.evaluation_function(reco)
    np.testing.assert_array_almost_equal(y3, y4)  # should be equal


def test_function_transform() -> None:
    func = functionlib.ArtificialFunction("sphere", 2, num_blocks=1, noise_level=0.1)
    output = func._transform(np.array([0.0, 0]))
    np.testing.assert_equal(output.shape, (1, 2))
    np.testing.assert_equal(len(output), 1)


def test_artificial_function_summary() -> None:
    func = functionlib.ArtificialFunction("sphere", 5)
    testing.assert_set_equal(func.descriptors.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(func.descriptors["function_class"], "ArtificialFunction")


def test_functionlib_copy() -> None:
    func = functionlib.ArtificialFunction("sphere", 5, noise_level=0.2, num_blocks=4)
    func2 = func.copy()
    assert func.equivalent_to(func2)
    assert func._parameters["noise_level"] == func2._parameters["noise_level"]
    assert func is not func2


def test_compute_pseudotime() -> None:
    x = np.array([2.0, 2])
    func = functionlib.ArtificialFunction("sphere", 2)
    np.testing.assert_equal(func.compute_pseudotime(((x,), {}), 3), 1.0)
    np.random.seed(12)
    func = functionlib.ArtificialFunction("DelayedSphere", 2)
    np.testing.assert_almost_equal(func.compute_pseudotime(((x,), {}), 3), 0.00025003021607278633)
    # check minimum
    np.random.seed(None)
    func = functionlib.ArtificialFunction("DelayedSphere", 2)
    func([0, 0])  # trigger init
    x = func.transform_var._transforms[0].translation
    np.testing.assert_equal(func(x), 0)
    np.testing.assert_equal(func.compute_pseudotime(((x,), {}), 0), 0)


@testing.parametrized(
    no_noise=(2, False, False, False),
    noise=(2, True, False, True),
    noise_dissymmetry_pos=(2, True, True, False),  # no noise on right side
    noise_dissymmetry_neg=(-2, True, True, True),
    no_noise_with_dissymmetry_neg=(-2, False, True, False),
)
def test_noisy_call(x: int, noise: bool, noise_dissymmetry: bool, expect_noisy: bool) -> None:
    fx = functionlib._noisy_call(
        x=np.array([x]),
        transf=np.tanh,
        func=lambda y: np.arctanh(y)[0],  # type: ignore
        noise_level=float(noise),
        noise_dissymmetry=noise_dissymmetry,
        random_state=np.random.RandomState(0),
    )
    assert not np.isnan(fx)  # noise addition should not get out of function domain
    if expect_noisy:
        np.testing.assert_raises(AssertionError, np.testing.assert_almost_equal, fx, x, decimal=8)
    else:
        np.testing.assert_almost_equal(fx, x, decimal=8)


@pytest.mark.parametrize("independent_sigma", [True, False])  # type: ignore
@pytest.mark.parametrize("mutable_sigma", [True, False])  # type: ignore
def test_far_optimum_function(independent_sigma: bool, mutable_sigma: bool) -> None:
    func = functionlib.FarOptimumFunction(
        independent_sigma=independent_sigma, mutable_sigma=mutable_sigma
    ).copy()
    param = func.parametrization.spawn_child()
    assert isinstance(param, p.Array)
    assert isinstance(param.sigma, p.Array) == mutable_sigma
    assert param.sigma.value.size == (1 + independent_sigma)  # type: ignore
    param.mutate()
    new_val = param.sigma.value
    assert bool(np.sum(np.abs(new_val - func.parametrization.sigma.value))) == mutable_sigma  # type: ignore
    if independent_sigma and mutable_sigma:
        assert new_val[0] != new_val[1]  # type: ignore


def test_far_optimum_function_cases() -> None:
    cases = list(functionlib.FarOptimumFunction.itercases())
    assert len(cases) == 48
