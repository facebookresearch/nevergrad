# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
import sys
from unittest import SkipTest
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.common import testing
from . import base
from .optimizerlib import registry


def long_name(s: str):
    return len(s.replace("DiscreteOnePlusOne", "D1+1")) > 10


# decorators to be used when testing on Windows is unecessary
# or cumbersome
skip_win_perf = pytest.mark.skipif(
    sys.platform == "win32", reason="Slow, and no need to test performance on all platforms"
)


def suggestable(name: str) -> bool:
    # Some methods are not good with suggestions.
    keywords = ["TBPSA", "BO", "EMNA", "EDA", "BO", "Stupid", "Pymoo", "GOMEA"]
    return not any(x in name for x in keywords)


def suggestion_testing(
    name: str,
    instrumentation: ng.p.Array,
    suggestion: np.ndarray,
    budget: int,
    objective_function: tp.Callable[..., tp.Any],
    optimum: tp.Optional[np.ndarray] = None,
    threshold: tp.Optional[float] = None,
):
    optimizer_cls = registry[name]
    optim = optimizer_cls(instrumentation, budget)
    if optimum is None:
        optimum = suggestion
    optim.suggest(suggestion)
    optim.minimize(objective_function)
    if threshold is not None:
        assert (
            objective_function(optim.recommend().value) < threshold
        ), f"{name} proposes {optim.recommend().value} instead of {optimum} (threshold={threshold})"
        return
    assert np.all(
        optim.recommend().value == optimum
    ), f"{name} proposes {optim.recommend().value} instead of {optimum}"


@skip_win_perf  # type: ignore
@pytest.mark.parametrize("name", [r for r in registry if suggestable(r)])  # type: ignore
def test_suggest_optimizers(name: str) -> None:
    """Checks that each optimizer is able to converge when optimum is given"""

    if sum([ord(c) for c in name]) % 4 > 0 and name not in ["CMA", "PSO", "DE"]:
        raise SkipTest("Too expensive: we randomly skip 3/4 of these tests.")

    instrum = ng.p.Array(shape=(100,)).set_bounds(0.0, 1.0)
    instrum.set_integer_casting()
    suggestion = np.asarray([0] * 17 + [1] * 17 + [0] * 66)  # The optimum is the suggestion.
    target = lambda x: 0 if np.all(np.asarray(x, dtype=int) == suggestion) else 1
    suggestion_testing(name, instrum, suggestion, 7, target)


def good_at_suggest(name: str) -> bool:
    keywords = [
        "Noisy",
        "Optimistic",
        "DiscreteDE",
        "Multi",
        "Anisotropic",
        "BSO",
        "GOMEA",
        "Sparse",
        "Adaptive",
        "Doerr",
        "Recombining",
        "SA",
        "Lognormal",
        "PortfolioDiscreteOne",
        "FastGADiscreteOne",
    ]
    return not any(k in name for k in keywords)


@skip_win_perf  # type: ignore
@pytest.mark.parametrize("name", [r for r in registry if "iscre" in r and "Smooth" not in r and good_at_suggest(r) and r != "DiscreteOnePlusOne" and ("Lengler" not in r or "LenglerOne" in r)])  # type: ignore
def test_harder_suggest_optimizers(name: str) -> None:
    """Checks that discrete optimizers are good when a suggestion is nearby."""
    if long_name(name):
        return
    if "OLN" in name:
        return
    instrum = ng.p.Array(shape=(100,)).set_bounds(0.0, 1.0)
    instrum.set_integer_casting()
    optimum = np.asarray([0] * 17 + [1] * 17 + [0] * 66)
    target = lambda x: min(3, np.sum((np.asarray(x, dtype=int) - optimum) ** 2))
    suggestion = np.asarray([0] * 17 + [1] * 16 + [0] * 67)
    suggestion_testing(name, instrum, suggestion, 1500 + (1000 if "Lengler" in name else 0), target, optimum)


@skip_win_perf  # type: ignore
def test_harder_continuous_suggest_optimizers() -> None:
    """Checks that somes optimizer can converge when provided with a good suggestion."""
    instrum = ng.p.Array(shape=(100,)).set_bounds(0.0, 1.0)
    optimum = np.asarray([0] * 17 + [1] * 17 + [0] * 66)
    target = lambda x: min(2.0, np.sum((x - optimum) ** 2))
    suggestion = np.asarray([0] * 17 + [1] * 16 + [0] * 67)
    suggestion_testing("NGOpt", instrum, suggestion, 1500, target, optimum, threshold=0.9)


@testing.suppress_nevergrad_warnings()
@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers_suggest(name: str) -> None:  # pylint: disable=redefined-outer-name
    optimizer = registry[name](parametrization=4, budget=2)
    optimizer.suggest(np.array([12.0] * 4))
    candidate = optimizer.ask()
    try:
        optimizer.tell(candidate, 12)
        # The optimizer should recommend its suggestion, except for a few optimization methods:
        if name not in ["SPSA", "TBPSA", "StupidRandom"]:
            np.testing.assert_array_almost_equal(optimizer.provide_recommendation().value, [12.0] * 4)
    except base.errors.TellNotAskedNotSupportedError:
        pass
