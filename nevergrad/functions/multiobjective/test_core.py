# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import core


def test_multiobjective_function() -> None:
    mfunc = core.MultiobjectiveFunction(lambda x: x, (100, 100))  # type: ignore
    tuples = [(110, 110), (110, 90), (80, 80), (50, 50), (50, 50), (80, 80), (30, 60), (60, 30)]
    values = []
    for tup in tuples:
        values.append(mfunc(tup))
    expected = [0, float('inf'), -400, -2500.0, -2500.0, -2500.0, -3300.0, -4100.0]
    # TODO (80, 80) should yield -2470, and values above the upper bounds could yield positive numbers
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p[0][0] for p in mfunc.pareto_front]
    expected_front = [(50, 50), (30, 60), (60, 30)]
    assert front == expected_front, f"Expected {expected_front} but got {front}"
