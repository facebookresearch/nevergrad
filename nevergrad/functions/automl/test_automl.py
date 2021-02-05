# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import nevergrad as ng
import numpy as np
from .core import AutoSKlearnBenchmark

def test_parametrization():
    func = AutoSKlearnBenchmark(openml_task_id=3, cv=3, time_budget_per_run=360,
                 memory_limit=7000, scoring_func="balanced_accuracy",
                 random_state=42)
    optimizer = ng.optimizers.BO(parametrization=func.parametrization, budget=10)
    recommendation = optimizer.minimize(func, verbosity=2)
    print(recommendation.value)
