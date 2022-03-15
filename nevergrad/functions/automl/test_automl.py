# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import nevergrad as ng

from .core import AutoSKlearnBenchmark


def test_parametrization():
    func = AutoSKlearnBenchmark(
        openml_task_id=3,
        cv=3,
        time_budget_per_run=60,
        memory_limit=2000,
        scoring_func="balanced_accuracy",
        random_state=42,
    )
    optimizer = ng.optimizers.RandomSearch(parametrization=func.parametrization, budget=3)
    optimizer.minimize(func, verbosity=2)


def test_function():
    func = AutoSKlearnBenchmark(
        openml_task_id=3,
        cv=3,
        time_budget_per_run=360,
        memory_limit=7000,
        scoring_func="balanced_accuracy",
        random_state=42,
    )
    for _ in range(2):
        is_valid = False
        while not is_valid:
            cand = func.parametrization.sample()
            is_valid = cand.satisfies_constraints()
        val = func(**cand.kwargs)
        assert (val >= 0) and (val <= 1)
