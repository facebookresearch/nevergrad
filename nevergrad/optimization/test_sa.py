# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

# import numpy as np
import sys

# from scipy import stats
# import nevergrad as ng


# decorators to be used when testing on Windows is unecessary
# or cumbersome
skip_win_perf = pytest.mark.skipif(
    sys.platform == "win32", reason="Slow, and no need to test performance on all platforms"
)


@skip_win_perf  # type: ignore
def test_sa() -> None:
    pass


#    num_tests = 77
#    for o in []:  # ["DiscreteOnePlusOne"]: dirty temporary hack.  # type: ignore
#        values = []
#        valuesT = []
#        for _ in range(num_tests):
#            dim = 30
#            arity = 3
#            budget = 57
#            domain = ng.p.TransitionChoice(range(arity), ordered=False, repetitions=dim)
#            optimum = np.random.randint(arity, size=dim)
#
#            def of(x):
#                return np.abs(np.abs(sum(x) - sum(optimum)) - 7.0)
#
#            recom = ng.optimizers.registry[o](domain, budget).minimize(of).value
#            recomT = ng.optimizers.registry["SA" + o + "Exp09"](domain, budget).minimize(of).value
#            values += [of(recom)]
#            valuesT += [of(recomT)]
#        pval = stats.mannwhitneyu(valuesT, values, alternative="less").pvalue
#        assert pval < 0.2, f"Simulated Annealing {o} fails the test: pval = {pval}."
