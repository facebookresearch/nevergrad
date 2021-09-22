# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import time
import collections
import typing as tp
import pytest
import numpy as np
from .optimizerlib import registry
from . import test_optimizerlib


KEY = "NEVERGRAD_SPECIAL_TESTS"
if not os.environ.get(KEY, ""):
    pytest.skip(f"These tests only run if {KEY} is set in the environment", allow_module_level=True)


@pytest.mark.parametrize("dimension", (2, 4, 7, 77))
@pytest.mark.parametrize("num_workers", (1,))
@pytest.mark.parametrize("scale", (4.0,))
@pytest.mark.parametrize("baseline", ["MetaModel", "CMA", "ECMA"])
@pytest.mark.parametrize("budget", [400, 4000])
@pytest.mark.parametrize("ellipsoid", [True, False])
def test_metamodel_sqp_chaining(
    dimension: int, num_workers: int, scale: float, budget: int, ellipsoid: bool, baseline: str
) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    target = test_optimizerlib.QuadFunction(scale=scale, ellipse=ellipsoid)
    baseline = baseline if dimension > 1 else "OnePlusOne"
    chaining = "ChainMetaModelSQP"

    # In both cases we compare MetaModel and CMA for a same given budget.
    # But we expect MetaModel to be clearly better only for a larger budget in the ellipsoid case.
    contextual_budget = budget if ellipsoid else 3 * budget
    contextual_budget *= 5 * int(max(1, np.sqrt(scale)))

    num_trials = 27
    successes = 0.0
    durations: tp.Dict[str, float] = collections.defaultdict(int)
    for _ in range(num_trials):
        if successes >= num_trials / 2:
            break
        # Let us run the comparison.
        recoms: tp.Dict[str, np.ndarray] = {}
        for name in (chaining, baseline):
            opt = registry[name](dimension, contextual_budget, num_workers=num_workers)
            t0 = time.time()
            recoms[name] = opt.minimize(target).value
            durations[name] += time.time() - t0

        if target(recoms[baseline]) < target(recoms[chaining]):
            successes += 1
        if target(recoms[baseline]) == target(recoms[chaining]):
            successes += 0.5

    if successes <= num_trials // 2:
        print(
            f"ChainMetaModelSQP fails ({successes}/{num_trials}) for d={dimension}, scale={scale}, "
            f"num_workers={num_workers}, ellipsoid={ellipsoid}, budget={budget}, vs {baseline}"
        )
        raise AssertionError("ChaingMetaModelSQP fails by performance.")
    print(
        f"ChainMetaModelSQP wins for d={dimension}, scale={scale}, num_workers={num_workers}, "
        f"ellipsoid={ellipsoid}, budget={budget}, vs {baseline}"
    )
    assert durations[chaining] < 7 * durations[baseline], "Computationally more than 7x more expensive."


@pytest.mark.parametrize("args", test_optimizerlib.get_metamodel_test_settings(special=True))
@pytest.mark.parametrize("baseline", ("CMA", "ECMA"))
def test_metamodel_special(baseline: str, args: tp.Tuple[tp.Any, ...]) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    kwargs = dict(zip(test_optimizerlib.META_TEST_ARGS, args))
    test_optimizerlib.check_metamodel(baseline=baseline, **kwargs)
