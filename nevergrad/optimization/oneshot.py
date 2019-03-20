# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import numpy as np
from scipy import stats
from ..common.typetools import ArrayLike
from ..instrumentation import Instrumentation
from . import sequences
from . import base

# # # # # classes of optimizers # # # # #


class OneShotOptimizer(base.Optimizer):
    # pylint: disable=abstract-method
    one_shot = True

# # # # # very basic baseline optimizers # # # # #


class _RandomSearch(OneShotOptimizer):

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._parameters = RandomSearchMaker()  # updated by the parametrized family

    def _internal_ask(self) -> ArrayLike:
        # pylint: disable=not-callable
        if self._parameters.middle_point and not self._num_ask:
            return np.zeros(self.dimension)  # type: ignore
        scale = self._parameters.scale
        if isinstance(scale, str) and scale == "random":
            scale = np.exp(np.random.normal(0., 1.) - 2.) / np.sqrt(self.dimension)
        point = np.random.standard_cauchy(self.dimension) if self._parameters.cauchy else np.random.normal(0, 1, self.dimension)
        return scale * point  # type: ignore

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self._parameters.stupid:
            return self._internal_ask()
        return super()._internal_provide_recommendation()


class RandomSearchMaker(base.ParametrizedFamily):
    """Provides random suggestions.

    Parameters
    ----------
    stupid: bool
        Provides a random recommendation instead of the best point so far (for baseline)
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    cauchy: bool
        use a Cauchy distribution instead of Gaussian distribution
    scale: float or "random"
        scalar for multiplying the suggested point values. If "random", this
        used a randomized pattern for the scale.
    """

    _optimizer_class = _RandomSearch
    one_shot = True

    # pylint: disable=unused-argument
    def __init__(self, *, middle_point: bool = False, stupid: bool = False,
                 cauchy: bool = False, scale: Union[float, str] = 1.) -> None:
        # keep all parameters and set initialize superclass for print
        assert isinstance(scale, (int, float)) or scale == "random"
        self.middle_point = middle_point
        self.stupid = stupid
        self.cauchy = cauchy
        self.scale = scale
        super().__init__()


Zero = RandomSearchMaker(scale=0.).with_name("Zero", register=True)
RandomSearch = RandomSearchMaker().with_name("RandomSearch", register=True)
RandomSearchPlusMiddlePoint = RandomSearchMaker(middle_point=True).with_name("RandomSearchPlusMiddlePoint", register=True)
LargerScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(
    middle_point=True, scale=500.).with_name("LargerScaleRandomSearchPlusMiddlePoint", register=True)
SmallScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(
    middle_point=True, scale=.01).with_name("SmallScaleRandomSearchPlusMiddlePoint", register=True)
StupidRandom = RandomSearchMaker(stupid=True).with_name("StupidRandom", register=True)
CauchyRandomSearch = RandomSearchMaker(cauchy=True).with_name("CauchyRandomSearch", register=True)
RandomScaleRandomSearch = RandomSearchMaker(
    scale="random", middle_point=True).with_name("RandomScaleRandomSearch", register=True)
RandomScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(
    scale="random", middle_point=True).with_name("RandomScaleRandomSearchPlusMiddlePoint", register=True)


class _SamplingSearch(OneShotOptimizer):

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._parameters = SamplingSearch()  # updated by the parametrized family
        self._sampler_instance: Optional[sequences.Sampler] = None
        self._rescaler: Optional[sequences.Rescaler] = None

    @property
    def sampler(self) -> sequences.Sampler:
        if self._sampler_instance is None:
            budget = None if self.budget is None else self.budget - self._parameters.middle_point
            samplers = {"Halton": sequences.HaltonSampler,
                        "Hammersley": sequences.HammersleySampler,
                        "LHS": sequences.LHSSampler,
                        }
            self._sampler_instance = samplers[self._parameters.sampler](self.dimension, budget, scrambling=self._parameters.scrambled)
            assert self._sampler_instance is not None
            if self._parameters.rescaled:
                self._rescaler = sequences.Rescaler(self.sampler)
                self._sampler_instance.reinitialize()  # sampler was consumed by the scaler
        return self._sampler_instance

    def _internal_ask(self) -> ArrayLike:
        # pylint: disable=not-callable
        if self._parameters.middle_point and not self._num_ask:
            return np.zeros(self.dimension)  # type: ignore
        sample = self.sampler()
        if self._rescaler is not None:
            sample = self._rescaler.apply(sample)
        return self._parameters.scale * (stats.cauchy.ppf if self._parameters.cauchy else stats.norm.ppf)(sample)  # type:ignore


class SamplingSearch(base.ParametrizedFamily):
    """This is a one-shot optimization method, hopefully better than random search
    by ensuring more uniformity.

    Parameters
    ----------
    sampler: str
        Choice of the sampler among "Halton", "Hammersley" and "LHS".
    scrambled: bool
        Adds scrambling to the search; much better in high dimension and rarely worse
        than the original search.
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    cauchy: bool
        use Cauchy inverse distribution instead of Gaussian when fitting points to real space
        (instead of box).
    scale: float or "random"
        scalar for multiplying the suggested point values.
    rescaled: bool
        rescales the sampling pattern to reach the boundaries.

    Notes
    -----
    - Halton is a low quality sampling method when the dimension is high; it is usually better
      to use Halton with scrambling.
    - When the budget is known in advance, it is also better to replace Halton by Hammersley.
      Basically the key difference with Halton is adding one coordinate evenly spaced
      (the discrepancy is better).
      budget, low discrepancy sequences (e.g. scrambled Hammersley) have a better discrepancy.
    - Reference: Halton 1964: Algorithm 247: Radical-inverse quasi-random point sequence, ACM, p. 701.
      adds scrambling to the Halton search; much better in high dimension and rarely worse
      than the original Halton search.
    - About Latin Hypercube Sampling (LHS):
      Though partially incremental versions exist, this implementation needs the budget in advance.
      This can be great in terms of discrepancy when the budget is not very high.
    """

    one_shot = True
    _optimizer_class = _SamplingSearch

    # pylint: disable=unused-argument
    def __init__(self, *, sampler: str = "Halton", scrambled: bool = False, middle_point: bool = False,
                 cauchy: bool = False, scale: float = 1., rescaled: bool = False) -> None:
        # keep all parameters and set initialize superclass for print
        self.sampler = sampler
        self.middle_point = middle_point
        self.scrambled = scrambled
        self.cauchy = cauchy
        self.scale = scale
        self.rescaled = rescaled
        super().__init__()


# pylint: disable=line-too-long
HaltonSearch = SamplingSearch().with_name("HaltonSearch", register=True)
HaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True).with_name("HaltonSearchPlusMiddlePoint", register=True)
LargeHaltonSearch = SamplingSearch(scale=100.).with_name("LargeHaltonSearch", register=True)
LargeScrHaltonSearch = SamplingSearch(scale=100., scrambled=True).with_name("LargeScrHaltonSearch", register=True)
LargeHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True).with_name("LargeHaltonSearchPlusMiddlePoint", register=True)
SmallHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True).with_name("SmallHaltonSearchPlusMiddlePoint", register=True)
ScrHaltonSearch = SamplingSearch(scrambled=True).with_name("ScrHaltonSearch", register=True)
ScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True).with_name("ScrHaltonSearchPlusMiddlePoint", register=True)
LargeScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, scrambled=True).with_name("LargeScrHaltonSearchPlusMiddlePoint", register=True)
SmallScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, scrambled=True).with_name("SmallScrHaltonSearchPlusMiddlePoint", register=True)
HammersleySearch = SamplingSearch(sampler="Hammersley").with_name("HammersleySearch", register=True)
HammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True).with_name("HammersleySearchPlusMiddlePoint", register=True)
LargeHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=100., sampler="Hammersley", middle_point=True).with_name("LargeHammersleySearchPlusMiddlePoint", register=True)
SmallHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=.01, sampler="Hammersley", middle_point=True).with_name("SmallHammersleySearchPlusMiddlePoint", register=True)
LargeScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=100., sampler="Hammersley", middle_point=True).with_name("LargeScrHammersleySearchPlusMiddlePoint", register=True)
SmallScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=.01, sampler="Hammersley", middle_point=True).with_name("SmallScrHammersleySearchPlusMiddlePoint", register=True)
ScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True).with_name("ScrHammersleySearchPlusMiddlePoint", register=True)
LargeHammersleySearch = SamplingSearch(scale=100., sampler="Hammersley").with_name("LargeHammersleySearch", register=True)
LargeScrHammersleySearch = SamplingSearch(
    scale=100., sampler="Hammersley", scrambled=True).with_name("LargeScrHammersleySearch", register=True)
ScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True).with_name("ScrHammersleySearch", register=True)
RescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, rescaled=True).with_name("RescaleScrHammersleySearch", register=True)
CauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True).with_name("CauchyScrHammersleySearch", register=True)
LHSSearch = SamplingSearch(sampler="LHS").with_name("LHSSearch", register=True)
CauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True).with_name("CauchyLHSSearch", register=True)
