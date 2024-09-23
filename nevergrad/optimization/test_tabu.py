# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
import sys
from scipy import stats
import nevergrad as ng
import nevergrad.common.typing as tp


# decorators to be used when testing on Windows is unecessary
# or cumbersome
skip_win_perf = pytest.mark.skipif(
    sys.platform == "win32", reason="Slow, and no need to test performance on all platforms"
)


@skip_win_perf  # type: ignore
def test_tabu() -> None:

    num_tests = 97
    for o in ["DiscreteOnePlusOne"]:
        values = []
        valuesT = []
        for _ in range(num_tests):
            dim = 4
            arity = 7
            budget = (arity**dim) // 50
            domain = ng.p.TransitionChoice(range(arity), ordered=False, repetitions=dim)
            optimum = np.random.randint(arity, size=dim)

            def of(x):
                return -np.sum((np.asarray(x) == optimum))

            recom = ng.optimizers.registry[o](domain, budget).minimize(of).value
            recomT = ng.optimizers.registry[o + "T"](domain, budget).minimize(of).value
            values += [of(recom)]
            valuesT += [of(recomT)]
        pval = stats.mannwhitneyu(valuesT, values, alternative="less").pvalue
        assert pval < 0.15, f"{o} fails the Tabu search test: pval = {pval}."


def summation(x: tp.ArrayLike) -> float:
    return sum(x)


@skip_win_perf  # type: ignore
def test_tabu_sum() -> None:

    num_tests = 147
    for o in ["DiscreteOnePlusOne"]:
        values = []
        valuesT = []
        for _ in range(num_tests):
            dim = 24
            arity = 3
            budget = 7
            domain = ng.p.TransitionChoice(range(arity), ordered=False, repetitions=dim)
            domain.tabu_congruence = summation
            optimum = np.random.randint(arity, size=dim)

            def of(x):
                return np.abs(sum(x) - sum(optimum))

            recom = ng.optimizers.registry[o](domain, budget).minimize(of).value
            recomT = ng.optimizers.registry[o + "T"](domain, budget).minimize(of).value
            values += [of(recom)]
            valuesT += [of(recomT)]
        pval = stats.mannwhitneyu(valuesT, values, alternative="less").pvalue
        assert pval < 0.1, f"{o} fails the Tabu search test: pval = {pval}."
