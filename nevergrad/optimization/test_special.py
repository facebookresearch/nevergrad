# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
import pytest
import nevergrad.common.typing as tp
from . import test_optimizerlib
from . import optimizerlib as optlib
import nevergrad as ng


KEY = "NEVERGRAD_SPECIAL_TESTS"
if not os.environ.get(KEY, ""):
    pytest.skip(f"These tests only run if {KEY} is set in the environment", allow_module_level=True)


class SimpleFitness:
    """Simple quadratic fitness function which can be used with dimension up to 4"""

    def __init__(self, x0: tp.ArrayLike, x1: tp.ArrayLike) -> None:
        self.x0 = np.array(x0, copy=True)
        self.x1 = np.array(np.exp(x1), copy=True)

    def __call__(self, x: tp.ArrayLike) -> float:
        assert len(self.x0) == len(x)
        return float(np.sum(self.x1 * np.cos(np.asarray(x) - self.x0) ** 2))


@pytest.mark.parametrize("dim", [2, 10, 40, 200])  # type: ignore
@pytest.mark.parametrize("bounded", [True])  # type: ignore
@pytest.mark.parametrize("discrete", [False])  # type: ignore
def test_performance_ngopt(dim: int, bounded: bool, discrete: bool) -> None:
    instrumentation = ng.p.Array(shape=(dim,))
    if dim > 40 and not discrete or dim <= 40 and discrete:
        return
    if bounded:
        instrumentation.set_bounds(lower=-12.0, upper=15.0)
    if discrete:
        instrumentation.set_integer_casting()
    algorithms = [optlib.NGOpt, optlib.OnePlusOne, optlib.CMA]
    if discrete:
        algorithms = [optlib.NGOpt, optlib.DiscreteOnePlusOne, optlib.DoubleFastGADiscreteOnePlusOne]
    num_tests = 7
    fitness = []
    for i in range(num_tests):
        target = np.random.normal(0.0, 1.0, size=dim)
        factors = np.random.normal(0.0, 7.0, size=dim)
        fitness += [SimpleFitness(target, factors)]
    result_tab = []
    for alg in algorithms:
        results = []
        for i in range(num_tests):
            result_for_this_fitness = []
            for budget_multiplier in [10, 100, 1000]:
                for num_workers in [1, 20]:
                    opt = alg(  # type: ignore
                        ng.p.Array(shape=(dim,)), budget=budget_multiplier * dim, num_workers=num_workers
                    )
                    recom = opt.minimize(fitness[i])
                    result_for_this_fitness += [fitness[i](recom.value)]
            results += result_for_this_fitness
        result_tab += [results]
        won_comparisons = [r < ngopt_res for (r, ngopt_res) in zip(result_tab[-1], result_tab[0])]
        assert (
            sum(won_comparisons) < len(won_comparisons) // 2
        ), f"alg={alg}, dim={dim}, budget_multiplier={budget_multiplier}, num_workers={num_workers}, bounded={bounded}, discrete={discrete}, result = {result_tab}"


@pytest.mark.parametrize("args", test_optimizerlib.get_metamodel_test_settings(special=True))
@pytest.mark.parametrize("baseline", ("CMA", "ECMA"))
def test_metamodel_special(baseline: str, args: tp.Tuple[tp.Any, ...]) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    kwargs = dict(zip(test_optimizerlib.META_TEST_ARGS, args))
    test_optimizerlib.check_metamodel(baseline=baseline, **kwargs)
