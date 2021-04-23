# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import nevergrad as ng
import numpy as np
import pytest
from nevergrad.benchmark.xpbase import Experiment
from .ngautosklearn import get_configuration

from .core import AutoSKlearnBenchmark


def test_parametrization():
    func = AutoSKlearnBenchmark(
        openml_task_id=3,
        cv=3,
        time_budget_per_run=360,
        memory_limit=7000,
        scoring_func="balanced_accuracy",
        random_state=42,
    )
    optimizer = ng.optimizers.RandomSearch(parametrization=func.parametrization, budget=3)
    optimizer.minimize(func, verbosity=2)


def test_bo_error():
    func = AutoSKlearnBenchmark(
        openml_task_id=3,
        cv=3,
        time_budget_per_run=360,
        memory_limit=7000,
        scoring_func="balanced_accuracy",
        random_state=42,
    )
    optimizer = ng.optimizers.BO(parametrization=func.parametrization, budget=10)
    for _ in range(5):
        config = optimizer.ask()
        optimizer.tell(config, 0.5)


def test_automl_experiment():
    func = AutoSKlearnBenchmark(
        openml_task_id=3,
        cv=3,
        time_budget_per_run=360,
        memory_limit=7000,
        scoring_func="balanced_accuracy",
        random_state=42,
    )
    exp = Experiment(func, "HyperOpt", 3, num_workers=1)
    exp.run()
