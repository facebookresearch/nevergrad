import os
import typing as tp
import pytest
import numpy as np
from .optimizerlib import registry
from . import test_optimizerlib


KEY = "NEVERGRAD_SPECIAL_TESTS"
if not os.environ.get(KEY, ""):
    pytest.skip(f"These tests only run if {KEY} is set in the environment", allow_module_level=True)


@pytest.mark.parametrize("dimension", (7,))
@pytest.mark.parametrize("num_workers", (1,))
@pytest.mark.parametrize("scale", (8.0,))
@pytest.mark.parametrize("baseline", ["MetaModel", "CMA", "ECMA"])
@pytest.mark.parametrize("budget", [200, 500])
@pytest.mark.parametrize("ellipsoid", [True, False])
def test_metamodel_sqp_chaining(
    dimension: int, num_workers: int, scale: float, budget: int, ellipsoid: bool, baseline: str
) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    target = test_optimizerlib.QuadFunction(scale=scale, ellipse=ellipsoid)

    # In both cases we compare MetaModel and CMA for a same given budget.
    # But we expect MetaModel to be clearly better only for a larger budget in the ellipsoid case.
    contextual_budget = budget if ellipsoid else 3 * budget
    contextual_budget *= 5 * int(max(1, np.sqrt(scale)))

    num_trials = 15
    successes = 0.0
    for _ in range(num_trials):
        if successes >= num_trials / 2:
            break
        # Let us run the comparison.
        recommendations: tp.List[np.ndarray] = []
        for name in ("ChainMetaModelSQP", baseline if dimension > 1 else "OnePlusOne"):
            opt = registry[name](dimension, contextual_budget, num_workers=num_workers)
            recommendations.append(opt.minimize(target).value)
        chaining_recom, default_recom = recommendations  # pylint: disable=unbalanced-tuple-unpacking

        if target(default_recom) < target(chaining_recom):
            successes += 1
        if target(default_recom) == target(chaining_recom):
            successes += 0.5

    if successes <= num_trials // 2:
        print(
            f"ChainMetaModelSQP fails ({successes}/{num_trials}) for d={dimension}, scale={scale}, num_workers={num_workers}, ellipsoid={ellipsoid}, budget={budget}, vs {baseline}"
        )
        assert False, "ChaingMetaModelSQP fails."
    print(
        f"ChainMetaModelSQP wins for d={dimension}, scale={scale}, num_workers={num_workers}, ellipsoid={ellipsoid}, budget={budget}, vs {baseline}"
    )


def get_tests_metamodel(seq: bool = False):
    tests_metamodel = [
        (2, 8, 1.0, 120, False),
        (2, 3, 8.0, 130, True),
        (5, 1, 1.0, 150, False),
    ]

    if not os.environ.get("CIRCLECI", False):
        # Interesting tests removed from CircleCI for flakiness (and we do stats when not on CircleCI):
        tests_metamodel += [
            (8, 27, 8.0, 380, True),
            (2, 1, 8.0, 120, True),
            (2, 3, 8.0, 70, False),
            (1, 1, 1.0, 20, True),
            (1, 3, 5.0, 20, False),
            (2, 3, 1.0, 70, True),
            (2, 1, 8.0, 40, False),
            (5, 3, 1.0, 225, True),
            (5, 1, 8.0, 150, False),
            (5, 3, 8.0, 500, True),
            (9, 27, 8.0, 700, True),
            (10, 27, 8.0, 400, False),
        ]
    if seq:
        for i in range(len(tests_metamodel)):
            d, _, s, b, e = tests_metamodel[i]
            tests_metamodel[i] = (d, 1, s, b, e)
    return tests_metamodel
