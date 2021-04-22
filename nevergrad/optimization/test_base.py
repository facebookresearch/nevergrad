# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from pathlib import Path
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import testing
import nevergrad as ng
from . import optimizerlib
from . import experimentalvariants as xpvariants
from . import base
from . import utils
from . import callbacks


class CounterFunction:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, value: tp.ArrayLike) -> float:
        assert len(value) == 1
        self.count += 1
        return float(value[0] - 1) ** 2


class LoggingOptimizer(base.Optimizer):
    def __init__(self, num_workers: int = 1) -> None:
        super().__init__(parametrization=1, budget=5, num_workers=num_workers)
        self.logs: tp.List[str] = []

    def _internal_ask(self) -> tp.ArrayLike:
        self.logs.append(f"s{self._num_ask}")  # s for suggest
        return np.array((float(self._num_ask),))

    # pylint: disable=unused-argument
    def _internal_tell(self, x: tp.ArrayLike, loss: float) -> None:
        self.logs.append(f"u{int(x[0])}")  # u for update


@testing.parametrized(
    w1_batch=(1, True, ["s0", "u0", "s1", "u1", "s2", "u2", "s3", "u3", "s4", "u4"]),
    w1_steady=(
        1,
        False,
        ["s0", "u0", "s1", "u1", "s2", "u2", "s3", "u3", "s4", "u4"],
    ),  # no difference (normal, since worker=1)
    w3_batch=(3, True, ["s0", "s1", "s2", "u0", "u1", "u2", "s3", "s4", "u3", "u4"]),
    w3_steady=(
        3,
        False,
        ["s0", "s1", "s2", "u0", "u1", "u2", "s3", "s4", "u3", "u4"],
    ),  # not really steady TODO change this behavior
    # w3_steady=(3, False, ['s0', 's1', 's2', 'u0', 's3', 'u1', 's4', 'u2', 'u3', 'u4']),  # This is what we would like
)
def test_batch_and_steady_optimization(
    num_workers: int, batch_mode: bool, expected: tp.List[tp.Tuple[str, float]]
) -> None:
    # tests the suggestion (s) and update (u) patterns
    # the w3_steady is unexpected. It is designed to be efficient with a non-sequential executor, but because
    # of it, it is acting like batch mode when sequential...
    optim = LoggingOptimizer(num_workers=num_workers)
    func = CounterFunction()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optim.minimize(func, verbosity=2, batch_mode=batch_mode)
    testing.printed_assert_equal(optim.logs, expected)


@testing.parametrized(
    int_val=(3, False),
    bool_val=(True, False),
    int32_val=(np.int32(3), False),
    int64_val=(np.int64(3), False),
    float64_val=(np.float64(3), False),
    float32_val=(np.float32(3), False),
    list_val=([3, 5], False),
    complex_val=(1j, True),
    object_val=(object(), True),
)
def test_tell_types(value: tp.Any, error: bool) -> None:
    optim = LoggingOptimizer(num_workers=1)
    x = optim.ask()
    if error:
        np.testing.assert_raises(TypeError, optim.tell, x, value)
    else:
        optim.tell(x, value)


def test_base_optimizer() -> None:
    zeroptim = xpvariants.Zero(parametrization=2, budget=4, num_workers=1)
    # add descriptor to replicate old behavior, returning pessimistic best
    zeroptim.parametrization.descriptors.deterministic_function = False
    assert not zeroptim.parametrization.function.deterministic
    representation = repr(zeroptim)
    expected = "parametrization=Array{(2,)}"
    assert expected in representation, f"Unexpected representation: {representation}"
    np.testing.assert_equal(zeroptim.ask().value, [0, 0])
    zeroptim.tell(zeroptim.parametrization.spawn_child().set_standardized_data([0.0, 0]), 0)
    zeroptim.tell(zeroptim.parametrization.spawn_child().set_standardized_data([1.0, 1]), 1)
    np.testing.assert_equal(zeroptim.provide_recommendation().value, [0, 0])
    # check that the best value is updated if a second evaluation is not as good
    zeroptim.tell(zeroptim.parametrization.spawn_child().set_standardized_data([0.0, 0]), 10)
    zeroptim.tell(zeroptim.parametrization.spawn_child().set_standardized_data([1.0, 1]), 1)
    np.testing.assert_equal(zeroptim.provide_recommendation().value, [1, 1])
    np.testing.assert_equal(zeroptim._num_ask, 1)
    # check suggest
    zeroptim.suggest([12, 12])
    np.testing.assert_array_equal(zeroptim.ask().args[0], [12, 12])
    np.testing.assert_array_equal(zeroptim.ask().args[0], [0, 0])


def test_optimize_and_dump(tmp_path: Path) -> None:
    optimizer = optimizerlib.OnePlusOne(parametrization=1, budget=100, num_workers=5)
    optimizer.register_callback(
        "tell", callbacks.OptimizationPrinter(print_interval_tells=10, print_interval_seconds=0.1)
    )
    func = CounterFunction()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimizer.minimize(func, verbosity=2)
    np.testing.assert_almost_equal(result.value[0], 1, decimal=2)
    np.testing.assert_equal(func.count, 100)
    # pickling
    filepath = tmp_path / "dump_test.pkl"
    optimizer.dump(filepath)
    optimizer2 = optimizerlib.OnePlusOne.load(filepath)
    np.testing.assert_almost_equal(optimizer2.provide_recommendation().value[0], 1, decimal=2)


def test_compare() -> None:
    optimizer = optimizerlib.CMA(parametrization=3, budget=1000, num_workers=5)
    optimizerlib.addCompare(optimizer)
    for _ in range(1000):  # TODO make faster test
        x: tp.List[tp.Any] = []
        for _ in range(6):
            x += [optimizer.ask()]
        winners = sorted(x, key=lambda x_: np.linalg.norm(x_.value - np.array((1.0, 1.0, 1.0))))
        optimizer.compare(winners[:3], winners[3:])  # type: ignore
    result = optimizer.provide_recommendation()
    print(result)
    np.testing.assert_almost_equal(result.value[0], 1.0, decimal=2)


def test_naming() -> None:
    optf = optimizerlib.RandomSearchMaker(stupid=True)
    opt = optf(parametrization=2, budget=4, num_workers=1)
    instru_str = "Array{(2,)}"
    np.testing.assert_equal(
        repr(opt),
        f"Instance of RandomSearchMaker(stupid=True)(parametrization={instru_str}, budget=4, num_workers=1)",
    )
    optf.set_name("BlubluOptimizer", register=True)
    opt = base.registry["BlubluOptimizer"](parametrization=2, budget=4, num_workers=1)
    np.testing.assert_equal(
        repr(opt), f"Instance of BlubluOptimizer(parametrization={instru_str}, budget=4, num_workers=1)"
    )


class MinStorageFunc:
    """Stores the minimum value obtained so far"""

    def __init__(self) -> None:
        self.min_loss = float("inf")

    def __call__(self, score: int) -> float:
        self.min_loss = min(score, self.min_loss)
        return score


def test_recommendation_correct() -> None:
    # Run this several times to debug:
    # pytest nevergrad/optimization/test_base.py::test_recommendation_correct --count=20 --exitfirst
    func = MinStorageFunc()
    choice_size = 20
    param = ng.p.Choice(range(choice_size)).set_name(f"Choice{choice_size}")
    optimizer = optimizerlib.OnePlusOne(parametrization=param, budget=300, num_workers=1)
    recommendation = optimizer.minimize(func)
    assert func.min_loss == recommendation.value


def constant(x: np.ndarray) -> float:  # pylint: disable=unused-argument
    return 12.0


def test_pruning_calls() -> None:
    opt = ng.optimizers.CMA(50, budget=2000)
    # worst case scenario for pruning is constant:
    # it should not keep everything or that will make computation time explode
    opt.minimize(constant)
    assert isinstance(opt.pruning, utils.Pruning)
    assert opt.pruning._num_prunings < 4
