# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import nevergrad as ng
from nevergrad.functions import MultiobjectiveFunction
from . import core


def test_multiobjective_function() -> None:
    mfunc = core.MultiobjectiveFunction(lambda x: x, (100, 100))  # type: ignore
    tuples = [(110, 110),     # -0 + distance
              (110, 90),      # -0 + distance
              (80, 80),       # -400 + distance
              (50, 50),       # -2500 + distance
              (50, 50),       # -2500 + distance
              (80, 80),       # -2500 + distance --> -2470
              (30, 60),       # [30,50]x[60,100] + [50,100]x[50,100] --> -2500 -800 = -3300 
              (60, 30)]       # [30,50]x[60,100] + [50,100]x[50,100] + [60,100]x[30,50] --> -2500 -800 -800= -4100 
    values = []
    for tup in tuples:
        values.append(mfunc(tup))
    expected = [10, 10, -400, -2500.0, -2500.0, -2470.0, -3300.0, -4100.0]
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p[0][0] for p in mfunc.pareto_front]
    expected_front = [(50, 50), (30, 60), (60, 30)]
    assert front == expected_front, f"Expected {expected_front} but got {front}"


def test_readme_example() -> None:
    f = MultiobjectiveFunction(multiobjective_function=lambda x: (x[0]**2, x[1]**2), upper_bounds=[2.5, 2.5])
    optimizer = ng.optimizers.CMA(instrumentation=3, budget=100)  # 3 is the dimension, 100 is the budget.
    optimizer.minimize(f)
    # The function embeds its Pareto-front:
    assert len(f.pareto_front) > 1
