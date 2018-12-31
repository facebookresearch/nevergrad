# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple
from scipy import stats
import numpy as np
from ..common.typetools import ArrayLike
from . import sequences
from . import base
from .base import registry

# # # # # classes of optimizers # # # # #


class OneShotOptimizer(base.Optimizer):
    # pylint: disable=abstract-method
    one_shot = True


# # # # # very basic baseline optimizers # # # # #


@registry.register
class Zero(OneShotOptimizer):
    """Always returns 0 (including for the recommendation)
    """

    def _internal_ask(self) -> Tuple[float, ...]:
        return tuple([0] * self.dimension)


@registry.register
class RandomSearch(OneShotOptimizer):
    """Provides random suggestions, and a recommendation
    based on the best returned fitness values.
    Use StupidRandom instead if you would rather the recommendation
    should not be based on past fitness values.
    """

    def _internal_ask(self) -> base.ArrayLike:
        return np.random.normal(0, 1, self.dimension)


@registry.register
class CauchyRandomSearch(OneShotOptimizer):
    """Provides random suggestions, and a recommendation
    based on the best returned fitness values.
    Uses Cauchy distribution.
    """

    def _internal_ask(self) -> base.ArrayLike:
        return np.random.standard_cauchy(self.dimension)


@registry.register
class StupidRandom(RandomSearch):
    """Provides random suggestions, and a random recommendation
    which is *not* based on past fitness values.
    Use RandomSearch instead if you would rather the recommendation
    be based on past fitness values.
    """

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return np.random.normal(0, 1, self.dimension)


# # # # # implementations # # # # #


@registry.register
class HaltonSearch(OneShotOptimizer):
    """Halton low-discrepancy search.

    This is a one-shot optimization method, hopefully better than random search
    by ensuring more uniformity.
    However, Halton is a low quality sampling method when the dimension is high;
    it is usually better to use Halton with scrambling.
    When the budget is known in advance, it is also better to replace Halton by Hammersley.
    Reference: Halton 1964: Algorithm 247: Radical-inverse quasi-random point sequence, ACM, p. 701.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        return stats.norm.ppf(self.sampler())


@registry.register
class ScrHaltonSearch(OneShotOptimizer):
    """Scrambled Halton search.

    Adds scrambling to the Halton search; much better in high dimension and rarely worse
    than the original Halton search.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        return stats.norm.ppf(self.sampler())


@registry.register
class HammersleySearch(OneShotOptimizer):
    """Hammersley version of the Halton search.

    Basically the key difference with Halton is adding one coordinate evenly spaced.
    The discrepancy is better; but we need the budget in advance."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        return stats.norm.ppf(self.sampler())


@registry.register
class ScrHammersleySearch(OneShotOptimizer):
    """Scrambled Hammersley sequence.

    This combines Scrambled Halton and Hammersley.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        return stats.norm.ppf(self.sampler())


@registry.register
class CauchyScrHammersleySearch(OneShotOptimizer):
    """Scrambled Hammersley sequence.

    This combines Scrambled Halton and Hammersley.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        return stats.cauchy.ppf(self.sampler())


@registry.register
class LHSSearch(OneShotOptimizer):
    """Latin Hypercube Sampling.

    Though partially incremental versions exist, this implementation needs the budget in advance.
    This can be great in terms of discrepancy when the budget is not very high - for high
    budget, low discrepancy sequences (e.g. scrambled Hammersley) have a better discrepancy.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        assert self.budget is not None, "A budget must be provided"
        self.sampler = sequences.LHSSampler(self.dimension, budget=self.budget)

    def _internal_ask(self) -> ArrayLike:
        return stats.norm.ppf(self.sampler())


@registry.register
class CauchyLHSSearch(OneShotOptimizer):
    """Latin Hypercube Sampling.

    Though partially incremental versions exist, this implementation needs the budget in advance.
    This can be great in terms of discrepancy when the budget is not very high - for high
    budget, low discrepancy sequences (e.g. scrambled Hammersley) have a better discrepancy.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        assert self.budget is not None, "A budget must be provided"
        self.sampler = sequences.LHSSampler(self.dimension, budget=self.budget)

    def _internal_ask(self) -> ArrayLike:
        return stats.cauchy.ppf(self.sampler())


@registry.register
class RescaleScrHammersleySearch(OneShotOptimizer):
    """Rescaled version of scrambled Hammersley search.

    We need the budget in advance, and rescale each variable linearly for almost matching the bounds.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        assert self.budget is not None, "A budget must be provided"
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=self.budget)
        self.rescaler = sequences.Rescaler(self.sampler)
        self.sampler.reinitialize()
        self.iterator = iter(self.sampler)

    def _internal_ask(self) -> ArrayLike:
        return stats.norm.ppf(self.rescaler.apply(next(self.iterator)))


@registry.register
class LargeHaltonSearch(OneShotOptimizer):
    """Larger scale Halton search.

    This corresponds to Halton, but pushing points closer to boundaries. This is in case
    extreme values are more likely to be useful.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class LargeScrHaltonSearch(OneShotOptimizer):
    """Larger scale scrambled Halton search."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class LargeHammersleySearch(OneShotOptimizer):
    """Larger scale Hammersley search."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class LargeScrHammersleySearch(OneShotOptimizer):
    """Larger scale scrambled Hammersley search."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class HaltonSearchPlusMiddlePoint(OneShotOptimizer):
    """Halton search with an additional middle point.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return stats.norm.ppf(self.sampler())


@registry.register
class ScrHaltonSearchPlusMiddlePoint(OneShotOptimizer):
    """Scrambled Halton search with an additional middle point.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return stats.norm.ppf(self.sampler())


@registry.register
class HammersleySearchPlusMiddlePoint(OneShotOptimizer):
    """Hammersley search with an additional middle point.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return stats.norm.ppf(self.sampler())


@registry.register
class ScrHammersleySearchPlusMiddlePoint(OneShotOptimizer):
    """Scrambled Hammersley search with an additional middle point.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return stats.norm.ppf(self.sampler())

# In "Large" samplers, all points are multiplied by 100.


@registry.register
class LargeHaltonSearchPlusMiddlePoint(OneShotOptimizer):
    """Adding a middle point in the larger scale Halton search.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class LargeScrHaltonSearchPlusMiddlePoint(OneShotOptimizer):
    """Adding a middle point in the larger scale scrambled Halton search.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class LargeHammersleySearchPlusMiddlePoint(OneShotOptimizer):
    """Adding a middle point in the larger scale Hammersley search.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class LargeScrHammersleySearchPlusMiddlePoint(OneShotOptimizer):
    """Adding a middle point in the larger scale scrambled Hammersley search.

    The additional point is the very first one."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 100. * stats.norm.ppf(self.sampler())


@registry.register
class SmallHaltonSearchPlusMiddlePoint(OneShotOptimizer):
    """Exact opposite of the version with "Large" instead of "Small"."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 0.01 * stats.norm.ppf(self.sampler())


@registry.register
class SmallScrHaltonSearchPlusMiddlePoint(OneShotOptimizer):
    """Exact opposite of the version with "Large" instead of "Small"."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHaltonSampler(self.dimension)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 0.01 * stats.norm.ppf(self.sampler())


@registry.register
class SmallHammersleySearchPlusMiddlePoint(OneShotOptimizer):
    """Exact opposite of the version with "Large" instead of "Small"."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.HammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 0.01 * stats.norm.ppf(self.sampler())


@registry.register
class SmallScrHammersleySearchPlusMiddlePoint(OneShotOptimizer):
    """Exact opposite of the version with "Large" instead of "Small"."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 0.01 * stats.norm.ppf(self.sampler())


@registry.register
class RandomSearchPlusMiddlePoint(OneShotOptimizer):
    """Random search plus a middle point.

    The middle point is the very first one."""

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return np.random.normal(0, 1, self.dimension)


@registry.register
class SmallScaleRandomSearchPlusMiddlePoint(OneShotOptimizer):

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 0.01 * np.random.normal(0, 1, self.dimension)


@registry.register
class RandomScaleRandomSearchPlusMiddlePoint(OneShotOptimizer):

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return np.exp(np.random.normal(0., 1.) - 2.) * np.random.normal(0., 1. / np.sqrt(self.dimension), self.dimension)


@registry.register
class RandomScaleRandomSearch(OneShotOptimizer):

    def _internal_ask(self) -> ArrayLike:
        return np.exp(np.random.normal(0., 1.) - 2.) * np.random.normal(0., 1. / np.sqrt(self.dimension), self.dimension)


@registry.register
class LargerScaleRandomSearchPlusMiddlePoint(OneShotOptimizer):

    def _internal_ask(self) -> ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return 500. * np.random.normal(0, 1, self.dimension)
