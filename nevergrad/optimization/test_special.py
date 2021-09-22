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
            f"ChainMetaModelSQP fails ({successes}/{num_trials}) for d={dimension}, scale={scale}, "
            f"num_workers={num_workers}, ellipsoid={ellipsoid}, budget={budget}, vs {baseline}"
        )
        assert False, "ChaingMetaModelSQP fails."
    print(
        f"ChainMetaModelSQP wins for d={dimension}, scale={scale}, num_workers={num_workers}, "
        f"ellipsoid={ellipsoid}, budget={budget}, vs {baseline}"
    )


@pytest.mark.parametrize("args", test_optimizerlib.get_metamodel_test_settings(special=True))
@pytest.mark.parametrize("baseline", ("CMA", "ECMA"))
def test_metamodel_special(baseline: str, args: tp.Tuple[tp.Any, ...]) -> None:
    """The test can operate on the sphere or on an elliptic funciton."""
    kwargs = dict(zip(test_optimizerlib.META_TEST_ARGS, args))
    test_optimizerlib.check_metamodel(baseline=baseline, **kwargs)
