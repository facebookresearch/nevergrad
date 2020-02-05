# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union
import numpy as np
from scipy import stats
from ..common.typetools import ArrayLike
from . import sequences
from . import base
from .base import IntOrParameter
from . import utils

# In some cases we will need the average of the k best.


def avg_of_k_best(archive: utils.Archive[utils.Value]) -> ArrayLike:
    # Operator inspired by the work of Yann Chevaleyre, Laurent Meunier, Clement Royer, Olivier Teytaud.
    items = list(archive.items_as_arrays())
    dimension = len(items[0][0])
    k = min(len(archive) // 4, dimension)  # fteytaud heuristic.
    k = 1 if k < 1 else k
    # Wasted time.
    first_k_individuals = [k for k in sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("pessimistic"))[:k]]
    assert len(first_k_individuals) == k
    return np.array(sum(p[0] for p in first_k_individuals) / k)

# # # # # classes of optimizers # # # # #


class OneShotOptimizer(base.Optimizer):
    # pylint: disable=abstract-method
    one_shot = True

# Recentering or center-based counterparts of the original Nevergrad oneshot optimizers:
# - Quasi-opposite counterpart of a sampling = one sample out of 2 is the symmetric of the previous one,
#   multiplied by rand([0,1]).
# - Opposite counterpart of a sampling = one sample out of 2 is the symmetric of the previous one.
# - PlusMiddlePoint counterpart of a sampling: we add (0,0,...,0) as a first point.
#   Useful in high dim.
# - Some variants use a rescaling depending on the budget and the dimension.


# # # # # One-shot optimizers: all fitness evaluations are in parallel. # # # # #


class _RandomSearch(OneShotOptimizer):

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = RandomSearchMaker()  # updated by the parametrized family
        self._opposable_data: Optional[np.ndarray] = None

    def _internal_ask(self) -> ArrayLike:
        # pylint: disable=not-callable
        mode = self._parameters.opposition_mode
        if self._opposable_data is not None and mode is not None:
            data = self._opposable_data
            data *= -(self._rng.uniform(0.0, 1.0) if mode == "quasi" else 1.0)
            self._opposable_data = None
            return data
        if self._parameters.middle_point and not self._num_ask:
            self._opposable_data = np.zeros(self.dimension)
            return self._opposable_data  # type: ignore
        scale = self._parameters.scale
        if isinstance(scale, str) and scale == "auto":
            # Some variants use a rescaling depending on the budget and the dimension.
            scale = (1 + np.log(self.budget)) / (4 * np.log(self.dimension))
        if isinstance(scale, str) and scale == "random":
            scale = np.exp(self._rng.normal(0., 1.) - 2.) / np.sqrt(self.dimension)
        point = (self._rng.standard_cauchy(self.dimension) if self._parameters.cauchy
                 else self._rng.normal(0, 1, self.dimension))
        self._opposable_data = scale * point
        return self._opposable_data  # type: ignore

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self._parameters.stupid:
            return self._internal_ask()
        if self._parameters.recommendation_rule == "average_of_best":
            return avg_of_k_best(self.archive)
        return super()._internal_provide_recommendation()


class RandomSearchMaker(base.ParametrizedFamily):
    """Provides random suggestions.

    Parameters
    ----------
    stupid: bool
        Provides a random recommendation instead of the best point so far (for baseline)
    middle_point: bool
        enforces that the first suggested point (ask) is zero.
    opposition_mode: str or None
        symmetrizes exploration wrt the center: (e.g. https://ieeexplore.ieee.org/document/4424748)
             - full symmetry if "opposite"
             - random * symmetric if "quasi"
    cauchy: bool
        use a Cauchy distribution instead of Gaussian distribution
    scale: float or "random"
        scalar for multiplying the suggested point values, or string:
         - "random": uses a randomized pattern for the scale.
         - "auto": scales in function of dimension and budget (see XXX)
    recommendation_rule: str
        "average_of_best" or "pessimistic"; "pessimistic" is the default and implies selecting the pessimistic best.
    """

    _optimizer_class = _RandomSearch
    one_shot = True

    # pylint: disable=unused-argument
    def __init__(self, *, middle_point: bool = False, stupid: bool = False,
                 opposition_mode: Optional[str] = None,
                 cauchy: bool = False, scale: Union[float, str] = 1.,
                 recommendation_rule: str = "pessimistic") -> None:
        # keep all parameters and set initialize superclass for print
        assert opposition_mode is None or opposition_mode in ["quasi", "opposite"]
        assert isinstance(scale, (int, float)) or scale in ["auto", "random"]
        self.middle_point = middle_point
        self.opposition_mode = opposition_mode
        self.stupid = stupid
        self.recommendation_rule = recommendation_rule
        self.cauchy = cauchy
        self.scale = scale
        super().__init__()


Zero = RandomSearchMaker(scale=0.).with_name("Zero", register=True)
RandomSearch = RandomSearchMaker().with_name("RandomSearch", register=True)
QORandomSearch = RandomSearchMaker(opposition_mode="quasi").with_name("QORandomSearch", register=True)
ORandomSearch = RandomSearchMaker(opposition_mode="opposite").with_name("ORandomSearch", register=True)
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

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = SamplingSearch()  # updated by the parametrized family
        self._sampler_instance: Optional[sequences.Sampler] = None
        self._rescaler: Optional[sequences.Rescaler] = None
        self._opposable_data: Optional[np.ndarray] = None

    @property
    def sampler(self) -> sequences.Sampler:
        if self._sampler_instance is None:
            budget = None if self.budget is None else self.budget - self._parameters.middle_point
            samplers = {"Halton": sequences.HaltonSampler,
                        "Hammersley": sequences.HammersleySampler,
                        "LHS": sequences.LHSSampler,
                        }
            internal_budget = (budget + 1) // 2 if budget and (self._parameters == "quasi" or self._parameters == "opposite") else budget
            self._sampler_instance = samplers[self._parameters.sampler](
                self.dimension, internal_budget, scrambling=self._parameters.scrambled, random_state=self._rng)
            assert self._sampler_instance is not None
            if self._parameters.rescaled:
                self._rescaler = sequences.Rescaler(self.sampler)
                self._sampler_instance.reinitialize()  # sampler was consumed by the scaler
        return self._sampler_instance

    def _internal_ask(self) -> ArrayLike:
        # pylint: disable=not-callable
        if self._parameters.middle_point and not self._num_ask:
            return np.zeros(self.dimension)  # type: ignore
        mode = self._parameters.opposition_mode
        if self._opposable_data is not None and mode is not None:
            # weird mypy error, revealed as array, but not accepting substraction
            data = self._opposable_data
            data *= -(self._rng.uniform(0.0, 1.0) if mode == "quasi" else 1.0)
            self._opposable_data = None
            return data
        sample = self.sampler()
        if self._rescaler is not None:
            sample = self._rescaler.apply(sample)
        if self._parameters.autorescale:
            self._parameters.scale = (1 + np.log(self.budget)) / (4 * np.log(self.dimension))
        self._opposable_data = self._parameters.scale * (
            stats.cauchy.ppf if self._parameters.cauchy else stats.norm.ppf)(sample)
        assert self._opposable_data is not None
        return self._opposable_data

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self._parameters.recommendation_rule == "average_of_best":
            return avg_of_k_best(self.archive)
        return super()._internal_provide_recommendation()


# pylint: disable=too-many-instance-attributes
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
    recommendation_rule: str
        "average_of_best" or "pessimistic"; "pessimistic" is the default and implies selecting the pessimistic best.

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
                 opposition_mode: Optional[str] = None,
                 cauchy: bool = False, autorescale: bool = False, scale: float = 1., rescaled: bool = False,
                 recommendation_rule: str = "pessimistic") -> None:
        # keep all parameters and set initialize superclass for print
        self.sampler = sampler
        self.opposition_mode = opposition_mode
        self.middle_point = middle_point
        self.scrambled = scrambled
        self.cauchy = cauchy
        self.autorescale = autorescale
        self.scale = scale
        self.rescaled = rescaled
        self.recommendation_rule = recommendation_rule
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
QOScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                       opposition_mode="quasi").with_name("QOScrHammersleySearch", register=True)
OScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                      opposition_mode="opposite").with_name("OScrHammersleySearch", register=True)
RescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, rescaled=True).with_name("RescaleScrHammersleySearch", register=True)
CauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True).with_name("CauchyScrHammersleySearch", register=True)
LHSSearch = SamplingSearch(sampler="LHS").with_name("LHSSearch", register=True)
CauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True).with_name("CauchyLHSSearch", register=True)


AvgHaltonSearch = SamplingSearch(recommendation_rule="average_of_best").with_name("AvgHaltonSearch", register=True)
AvgHaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True, recommendation_rule="average_of_best").with_name(
    "AvgHaltonSearchPlusMiddlePoint", register=True)
AvgLargeHaltonSearch = SamplingSearch(scale=100., recommendation_rule="average_of_best").with_name("AvgLargeHaltonSearch", register=True)
AvgLargeScrHaltonSearch = SamplingSearch(scale=100., scrambled=True, recommendation_rule="average_of_best").with_name(
    "AvgLargeScrHaltonSearch", register=True)
AvgLargeHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, recommendation_rule="average_of_best").with_name("AvgLargeHaltonSearchPlusMiddlePoint", register=True)
AvgSmallHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, recommendation_rule="average_of_best").with_name("AvgSmallHaltonSearchPlusMiddlePoint", register=True)
AvgScrHaltonSearch = SamplingSearch(scrambled=True, recommendation_rule="average_of_best").with_name("AvgScrHaltonSearch", register=True)
AvgScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True, recommendation_rule="average_of_best").with_name("AvgScrHaltonSearchPlusMiddlePoint", register=True)
AvgLargeScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, scrambled=True, recommendation_rule="average_of_best").with_name("AvgLargeScrHaltonSearchPlusMiddlePoint", register=True)
AvgSmallScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, scrambled=True, recommendation_rule="average_of_best").with_name("AvgSmallScrHaltonSearchPlusMiddlePoint", register=True)
AvgHammersleySearch = SamplingSearch(sampler="Hammersley", recommendation_rule="average_of_best").with_name(
    "AvgHammersleySearch", register=True)
AvgHammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best").with_name("AvgHammersleySearchPlusMiddlePoint", register=True)
AvgLargeHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=100., sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best").with_name("AvgLargeHammersleySearchPlusMiddlePoint", register=True)
AvgSmallHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=.01, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best").with_name("AvgSmallHammersleySearchPlusMiddlePoint", register=True)
AvgLargeScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=100., sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best").with_name("AvgLargeScrHammersleySearchPlusMiddlePoint", register=True)
AvgSmallScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=.01, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best").with_name("AvgSmallScrHammersleySearchPlusMiddlePoint", register=True)
AvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best").with_name("AvgScrHammersleySearchPlusMiddlePoint", register=True)
AvgLargeHammersleySearch = SamplingSearch(scale=100., sampler="Hammersley",
                                          recommendation_rule="average_of_best").with_name("AvgLargeHammersleySearch", register=True)
AvgLargeScrHammersleySearch = SamplingSearch(
    scale=100., sampler="Hammersley", scrambled=True, recommendation_rule="average_of_best").with_name("AvgLargeScrHammersleySearch", register=True)
AvgScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                        recommendation_rule="average_of_best").with_name("AvgScrHammersleySearch", register=True)
AvgRescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, rescaled=True, recommendation_rule="average_of_best").with_name("AvgRescaleScrHammersleySearch", register=True)
AvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_best").with_name("AvgCauchyScrHammersleySearch", register=True)
AvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_best").with_name("AvgLHSSearch", register=True)
AvgCauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True, recommendation_rule="average_of_best").with_name(
    "AvgCauchyLHSSearch", register=True)
