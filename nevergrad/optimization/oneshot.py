# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import stats
from scipy.spatial import ConvexHull  # pylint: disable=no-name-in-module
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import sequences
from . import base
from .base import IntOrParameter
from . import utils

# In some cases we will need the average of the k best.


def convex_limit(struct_points: np.ndarray) -> int:
    """Given points in order from best to worst,
    Returns the length of the maximum initial segment of points such that quasiconvexity is verified."""
    points: tp.List[float] = []
    d = len(struct_points[0])
    if len(struct_points) < 2 * d + 2:
        return len(struct_points) // 2
    for i in range(0, min(2 * d + 2, len(struct_points)), 2):
        points += [struct_points[i]]
    hull = ConvexHull(points, incremental=True)
    k = len(points)
    for i in range(d + 1, len(points)):
        hull.add_points(points[i : (i + 1)])
        if i not in hull.vertices:
            k = i - 1
            break
    return k


def hull_center(points: np.ndarray, k: int) -> np.ndarray:
    hull = ConvexHull(points[:k])
    maxi = np.asarray(hull.vertices[0])
    mini = np.asarray(hull.vertices[0])
    for v in hull.vertices:
        maxi = np.maximum(np.asarray(v), maxi)
        mini = np.minimum(np.asarray(v), mini)
    return 0.5 * (maxi + mini)  # type: ignore


def avg_of_k_best(archive: utils.Archive[utils.MultiValue], method: str = "dimfourth") -> np.ndarray:
    """Operators inspired by the work of Yann Chevaleyre, Laurent Meunier, Clement Royer, Olivier Teytaud, Fabien Teytaud.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
        Provides a random recommendation instead of the best point so far (for baseline)
    method: str
        If dimfourth, we use the Fteytaud heuristic, i.e. k = min(len(archive) // 4, dimension)
        If exp, we use the Lmeunier method, i.e. k=max(1, len(archiv) // (2**dimension))
        If hull, we use the maximum k <= dimfourth-value, such that the function looks quasiconvex on the k best points.
    """
    items = list(archive.items_as_arrays())
    dimension = len(items[0][0])
    if method == "dimfourth":
        k = min(len(archive) // 4, dimension)  # fteytaud heuristic.
    elif method == "exp":
        k = max(1, int(len(archive) // (1.1 ** dimension)))
    elif method == "hull":
        k = convex_limit(
            np.concatenate(
                sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("pessimistic")), axis=0
            )
        )
        k = min(len(archive) // 4, min(k, int(len(archive) / (1.1 ** dimension))))
        # We might investigate the possibility to return the middle of the convex hull instead of averaging:
        # return hull_center(np.concatenate(sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("pessimistic")), axis=0), k)
    else:
        raise ValueError(f"{method} not implemented as a method for choosing k in avg_of_k_best.")
    k = 1 if k < 1 else int(k)
    # Wasted time.
    first_k_individuals = sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("pessimistic"))[:k]
    assert len(first_k_individuals) == k
    return np.array(sum(p[0] for p in first_k_individuals) / k)


# # # # # classes of optimizers # # # # #


class OneShotOptimizer(base.Optimizer):
    one_shot = True

    def _internal_ask_candidate(self) -> p.Parameter:
        out = self.parametrization.spawn_child()
        with p.helpers.deterministic_sampling(out):
            # the values will be sampled deterministically since there is no reason not to
            # with one shot algorithms
            out.set_standardized_data(self._internal_ask())
        return out


# Recentering or center-based counterparts of the original Nevergrad oneshot optimizers:
# - Quasi-opposite counterpart of a sampling = one sample out of 2 is the symmetric of the previous one,
#   multiplied by rand([0,1]).
# - Opposite counterpart of a sampling = one sample out of 2 is the symmetric of the previous one.
# - PlusMiddlePoint counterpart of a sampling: we add (0,0,...,0) as a first point.
#   Useful in high dim.
# - Some variants use a rescaling depending on the budget and the dimension.


# # # # # One-shot optimizers: all fitness evaluations are in parallel. # # # # #


# pylint: disable=too-many-arguments,too-many-instance-attributes
class _RandomSearch(OneShotOptimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        middle_point: bool = False,
        stupid: bool = False,
        opposition_mode: tp.Optional[str] = None,
        sampler: str = "parametrization",
        scale: tp.Union[float, str] = 1.0,
        recommendation_rule: str = "pessimistic",
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert opposition_mode is None or opposition_mode in ["quasi", "opposite"]
        assert isinstance(scale, (int, float)) or scale in ["auto", "random", "autotune"]
        self.middle_point = middle_point
        self.opposition_mode = opposition_mode
        self.stupid = stupid
        self.recommendation_rule = recommendation_rule
        self.scale = scale
        self.sampler = sampler
        self._opposable_data: tp.Optional[np.ndarray] = None

    def _internal_ask(self) -> tp.ArrayLike:
        # pylint: disable=not-callable
        mode = self.opposition_mode
        if self._opposable_data is not None and mode is not None:
            data = self._opposable_data
            data *= -(self._rng.uniform(0.0, 1.0) if mode == "quasi" else 1.0)
            self._opposable_data = None
            return data
        if self.middle_point and not self._num_ask:
            self._opposable_data = np.zeros(self.dimension)
            return self._opposable_data
        scale = self.scale
        if isinstance(scale, str) and scale == "auto":
            # Some variants use a rescaling depending on the budget and the dimension (1st version).
            scale = (1 + np.log(self.budget)) / (4 * np.log(self.dimension))
        if isinstance(scale, str) and scale == "autotune":
            scale = np.sqrt(np.log(self.budget) / self.dimension)
        if isinstance(scale, str) and scale == "random":
            scale = np.exp(self._rng.normal(0.0, 1.0) - 2.0) / np.sqrt(self.dimension)
        # sample the new point
        if self.sampler == "gaussian":
            point = self._rng.normal(0, 1, self.dimension)
        elif self.sampler == "cauchy":
            point = self._rng.standard_cauchy(self.dimension)
        elif self.sampler == "parametrization":
            point = self.parametrization.sample().get_standardized_data(reference=self.parametrization)
        else:
            raise ValueError("Unkwnown sampler {self.sampler}")
        self._opposable_data = scale * point
        return self._opposable_data  # type: ignore

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        if self.stupid:
            return self._internal_ask()
        elif self.archive:
            if self.recommendation_rule == "average_of_best":
                return avg_of_k_best(self.archive, "dimfourth")
            if self.recommendation_rule == "average_of_exp_best":
                return avg_of_k_best(self.archive, "exp")
            if self.recommendation_rule == "average_of_hull_best":
                return avg_of_k_best(self.archive, "hull")
        return None  # back to default


class RandomSearchMaker(base.ConfiguredOptimizer):
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
    sampler: str
        - parametrization: uses the default sample() method of the parametrization, which samples uniformly
          between bounds and a Gaussian otherwise
        - gaussian: uses a Gaussian distribution
        - cauchy: uses a Cauchy distribution
        use a Cauchy distribution instead of Gaussian distribution
    scale: float or "random"
        scalar for multiplying the suggested point values, or string:
         - "random": uses a randomized pattern for the scale.
         - "auto": scales in function of dimension and budget (version 1: sigma = (1+log(budget)) / (4log(dimension)) )
         - "autotune": scales in function of dimension and budget (version 2: sigma = sqrt(log(budget) / dimension) )
    recommendation_rule: str
        "average_of_best" or "pessimistic" or "average_of_exp_best"; "pessimistic" is
        the default and implies selecting the pessimistic best.
    """

    one_shot = True

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        middle_point: bool = False,
        stupid: bool = False,
        opposition_mode: tp.Optional[str] = None,
        sampler: str = "parametrization",
        scale: tp.Union[float, str] = 1.0,
        recommendation_rule: str = "pessimistic",
    ) -> None:
        assert sampler in ["gaussian", "cauchy", "parametrization"]
        super().__init__(_RandomSearch, locals())


RandomSearch = RandomSearchMaker().set_name("RandomSearch", register=True)
QORandomSearch = RandomSearchMaker(opposition_mode="quasi").set_name("QORandomSearch", register=True)
ORandomSearch = RandomSearchMaker(opposition_mode="opposite").set_name("ORandomSearch", register=True)
RandomSearchPlusMiddlePoint = RandomSearchMaker(middle_point=True).set_name(
    "RandomSearchPlusMiddlePoint", register=True
)


class _SamplingSearch(OneShotOptimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        sampler: str = "Halton",
        scrambled: bool = False,
        middle_point: bool = False,
        opposition_mode: tp.Optional[str] = None,
        cauchy: bool = False,
        autorescale: tp.Union[bool, str] = False,
        scale: float = 1.0,
        rescaled: bool = False,
        recommendation_rule: str = "pessimistic",
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._sampler_instance: tp.Optional[sequences.Sampler] = None
        self._rescaler: tp.Optional[sequences.Rescaler] = None
        self._opposable_data: tp.Optional[np.ndarray] = None
        self._sampler = sampler
        self.opposition_mode = opposition_mode
        self.middle_point = middle_point
        self.scrambled = scrambled
        self.cauchy = cauchy
        self.autorescale = autorescale
        self.scale = scale
        self.rescaled = rescaled
        self.recommendation_rule = recommendation_rule
        # rescale to the bounds if both are provided
        self._scaler = utils.BoundScaler(self.parametrization)

    @property
    def sampler(self) -> sequences.Sampler:
        if self._sampler_instance is None:
            budget = None if self.budget is None else self.budget - self.middle_point
            samplers = {
                "Halton": sequences.HaltonSampler,
                "Hammersley": sequences.HammersleySampler,
                "LHS": sequences.LHSSampler,
                "Random": sequences.RandomSampler,
            }
            internal_budget = (
                (budget + 1) // 2 if budget and (self.opposition_mode in ["quasi", "opposite"]) else budget
            )
            self._sampler_instance = samplers[self._sampler](
                self.dimension, internal_budget, scrambling=self.scrambled, random_state=self._rng
            )
            assert self._sampler_instance is not None
            if self.rescaled:
                self._rescaler = sequences.Rescaler(self.sampler)
                self._sampler_instance.reinitialize()  # sampler was consumed by the scaler
        return self._sampler_instance

    def _internal_ask(self) -> tp.ArrayLike:
        # pylint: disable=not-callable
        if self.middle_point and not self._num_ask:
            return np.zeros(self.dimension)
        mode = self.opposition_mode
        if self._opposable_data is not None and mode is not None:
            # weird mypy error, revealed as array, but not accepting substraction
            data = self._opposable_data
            data *= -(self._rng.uniform(0.0, 1.0) if mode == "quasi" else 1.0)
            self._opposable_data = None
            return data
        sample = self.sampler()
        if self._rescaler is not None:
            sample = self._rescaler.apply(sample)
        if self.autorescale is True or self.autorescale == "auto":
            self.scale = (1 + np.log(self.budget)) / (4 * np.log(self.dimension))
        if self.autorescale == "autotune":
            self.scale = np.sqrt(np.log(self.budget) / self.dimension)

        def transf(x: np.ndarray) -> np.ndarray:
            return self.scale * (stats.cauchy.ppf if self.cauchy else stats.norm.ppf)(x)  # type: ignore

        self._opposable_data = self._scaler.transform(sample, transf)
        assert self._opposable_data is not None
        return self._opposable_data

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        if self.archive and self.recommendation_rule == "average_of_best":
            return avg_of_k_best(self.archive)
        return None


# pylint: disable=too-many-instance-attributes
class SamplingSearch(base.ConfiguredOptimizer):
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
    rescaled: bool or str
        rescales the sampling pattern to reach the boundaries and/or applies automatic rescaling.
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

    # pylint: disable=unused-argument,too-many-arguments
    def __init__(
        self,
        *,
        sampler: str = "Halton",
        scrambled: bool = False,
        middle_point: bool = False,
        opposition_mode: tp.Optional[str] = None,
        cauchy: bool = False,
        autorescale: tp.Union[bool, str] = False,
        scale: float = 1.0,
        rescaled: bool = False,
        recommendation_rule: str = "pessimistic",
    ) -> None:
        super().__init__(_SamplingSearch, locals())


# pylint: disable=line-too-long
MetaRecentering = SamplingSearch(
    cauchy=False, autorescale=True, sampler="Hammersley", scrambled=True
).set_name("MetaRecentering", register=True)
MetaTuneRecentering = SamplingSearch(
    cauchy=False, autorescale="autotune", sampler="Hammersley", scrambled=True
).set_name("MetaTuneRecentering", register=True)
HAvgMetaRecentering = SamplingSearch(
    cauchy=False,
    autorescale=True,
    sampler="Hammersley",
    scrambled=True,
    recommendation_rule="average_of_hull_best",
).set_name("HAvgMetaRecentering", register=True)
AvgMetaRecenteringNoHull = SamplingSearch(
    cauchy=False,
    autorescale=True,
    sampler="Hammersley",
    scrambled=True,
    recommendation_rule="average_of_exp_best",
).set_name("AvgMetaRecenteringNoHull", register=True)

HaltonSearch = SamplingSearch().set_name("HaltonSearch", register=True)
HaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True).set_name(
    "HaltonSearchPlusMiddlePoint", register=True
)
LargeHaltonSearch = SamplingSearch(scale=100.0).set_name("LargeHaltonSearch", register=True)
ScrHaltonSearch = SamplingSearch(scrambled=True).set_name("ScrHaltonSearch", register=True)
ScrHaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True, scrambled=True).set_name(
    "ScrHaltonSearchPlusMiddlePoint", register=True
)
HammersleySearch = SamplingSearch(sampler="Hammersley").set_name("HammersleySearch", register=True)
HammersleySearchPlusMiddlePoint = SamplingSearch(sampler="Hammersley", middle_point=True).set_name(
    "HammersleySearchPlusMiddlePoint", register=True
)
ScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True
).set_name("ScrHammersleySearchPlusMiddlePoint", register=True)
ScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True).set_name(
    "ScrHammersleySearch", register=True
)
QOScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, opposition_mode="quasi"
).set_name("QOScrHammersleySearch", register=True)
OScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, opposition_mode="opposite"
).set_name("OScrHammersleySearch", register=True)
CauchyScrHammersleySearch = SamplingSearch(cauchy=True, sampler="Hammersley", scrambled=True).set_name(
    "CauchyScrHammersleySearch", register=True
)
LHSSearch = SamplingSearch(sampler="LHS").set_name("LHSSearch", register=True)
CauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True).set_name("CauchyLHSSearch", register=True)
