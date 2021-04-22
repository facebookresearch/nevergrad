# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from .hypervolume import HypervolumeIndicator


AUTO_BOUND = 15


# pylint: disable=too-many-instance-attributes
class HypervolumePareto:
    """Given several functions, and threshold on their values (above which solutions are pointless),
    this object can be used as a single-objective function, the minimization of which
    yields a solution to the original multiobjective problem.

    Parameters
    -----------
    upper_bounds: Tuple of float or np.ndarray
        upper_bounds[i] is a threshold above which x is pointless if function(x)[i] > upper_bounds[i].
    auto_bound: int
        if no upper bounds are provided, number of initial points used to estimate the upper bounds. Their
        loss will be 0 (except if they are uniformly worse than the previous points).
    seed: optional int or RandomState
        seed to use for selecting random subsamples of the pareto

    Notes
    -----
    - This function is not stationary!
    - The minimum value obtained for this objective function is -h,
      where h is the hypervolume of the Pareto front obtained, given upper_bounds as a reference point.
    - The callable keeps track of the pareto_front (see attribute paretor_front) and is therefor stateful.
      For this reason it cannot be distributed. A user can however call the multiobjective_function
      remotely, and aggregate locally. This is what happens in the "minimize" method of optimizers.
    """

    def __init__(
        self,
        upper_bounds: tp.Optional[tp.ArrayLike] = None,
        auto_bound: int = AUTO_BOUND,
        seed: tp.Optional[tp.Union[int, np.random.RandomState]] = None,
    ) -> None:
        self._auto_bound = 0
        self._upper_bounds = (
            np.array([-float("inf")]) if upper_bounds is None else np.array(upper_bounds, copy=False)
        )
        if upper_bounds is None:
            self._auto_bound = auto_bound
        self._rng = seed if isinstance(seed, np.random.RandomState) else np.random.RandomState(seed)
        self._pareto: tp.List[p.Parameter] = []
        self._best_volume = -float("Inf")
        self._hypervolume: tp.Optional[HypervolumeIndicator] = None
        self._pareto_needs_filtering = False

    @property
    def num_objectives(self) -> int:
        return self._upper_bounds.size

    @property
    def best_volume(self) -> float:
        return self._best_volume

    def _add_to_pareto(self, parameter: p.Parameter) -> None:
        self._pareto.append(parameter)
        self._pareto_needs_filtering = True

    def extend(self, parameters: tp.Sequence[p.Parameter]) -> float:
        output = 0.0
        for param in parameters:
            output = self.add(param)
        return output

    def add(self, parameter: p.Parameter) -> float:
        """Given parameters and the multiobjective loss, this computes the hypervolume
        and update the state of the function with new points if it belongs to the pareto front
        """
        if not isinstance(parameter, p.Parameter):
            raise TypeError(
                f"{self.__class__.__name__}.add should receive a ng.p.Parameter, but got: {parameter}."
            )
        losses = parameter.losses
        if not isinstance(losses, np.ndarray):
            raise TypeError(
                f"Parameter should have multivalue as losses, but parameter.losses={losses} ({type(losses)})."
            )
        if self._auto_bound > 0:
            self._auto_bound -= 1
            if (self._upper_bounds > -float("inf")).all() and (losses > self._upper_bounds).all():
                return float("inf")  # Avoid uniformly worst points
            self._upper_bounds = np.maximum(self._upper_bounds, losses)
            self._add_to_pareto(parameter)
            return 0.0
        if self._hypervolume is None:
            self._hypervolume = HypervolumeIndicator(self._upper_bounds)
        # get rid of points over the upper bounds
        if (losses - self._upper_bounds > 0).any():
            loss = -float(np.sum(np.maximum(0, losses - self._upper_bounds)))
            if loss > self._best_volume:
                self._best_volume = loss
            if self._best_volume < 0:
                self._add_to_pareto(parameter)
            return -loss
        # We compute the hypervolume
        new_volume = self._hypervolume.compute([pa.losses for pa in self._pareto] + [losses])
        if new_volume > self._best_volume:
            # This point is good! Let us give him a great mono-fitness value.
            self._best_volume = new_volume
            self._add_to_pareto(parameter)
            return -new_volume
        else:
            # This point is not on the front
            # First we prune.
            distance_to_pareto = float("Inf")
            for param in self.pareto_front():
                stored_losses = param.losses
                # TODO the following is probably not good at all:
                # -> +inf if no point is strictly better (but lower if it is)
                if (stored_losses <= losses).all():
                    distance_to_pareto = min(distance_to_pareto, min(losses - stored_losses))
            assert distance_to_pareto >= 0
            return -new_volume + distance_to_pareto

    def _filter_pareto_front(self) -> None:
        """Filters the Pareto front"""
        new_pareto: tp.List[p.Parameter] = []
        for param in self._pareto:  # quadratic :(
            should_be_added = True
            for other in self._pareto:
                if (other.losses <= param.losses).all() and (other.losses < param.losses).any():
                    should_be_added = False
                    break
            if should_be_added:
                new_pareto.append(param)
        self._pareto = new_pareto
        self._pareto_needs_filtering = False

    # pylint: disable=too-many-branches
    def pareto_front(
        self, size: tp.Optional[int] = None, subset: str = "random", subset_tentatives: int = 12
    ) -> tp.List[p.Parameter]:
        """Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "EPS", "domain-covering", "hypervolume")
            EPS is the epsilon indicator described e.g.
                here: https://hal.archives-ouvertes.fr/hal-01159961v2/document
        subset_tentatives: int
            number of random tentatives for finding a better subset

        Returns
        --------
        list
            the list of Parameter of the pareto front
        """
        if self._pareto_needs_filtering:
            self._filter_pareto_front()
        if size is None or size >= len(self._pareto):  # No limit: we return the full set.
            return self._pareto
        if subset == "random":
            return self._rng.choice(self._pareto, size).tolist()  # type: ignore
        tentatives = [self._rng.choice(self._pareto, size).tolist() for _ in range(subset_tentatives)]
        if self._hypervolume is None:
            raise RuntimeError("Hypervolume not initialized, not supported")  # TODO fix
        hypervolume = self._hypervolume
        scores: tp.List[float] = []
        for tentative in tentatives:
            if subset == "hypervolume":
                scores += [-hypervolume.compute([pa.losses for pa in tentative])]
            else:
                score: float = 0.0
                for v in self._pareto:
                    best_score = float("inf") if subset != "EPS" else 0.0
                    for pa in tentative:
                        if subset == "loss-covering":  # Equivalent to IGD.
                            best_score = min(best_score, np.linalg.norm(pa.losses - v.losses))
                        elif subset == "EPS":  # Cone Epsilon-Dominance.
                            best_score = min(best_score, max(pa.losses - v.losses))
                        elif subset == "domain-covering":
                            best_score = min(
                                best_score, np.linalg.norm(pa.get_standardized_data(reference=v))
                            )  # TODO verify
                        else:
                            raise ValueError(f'Unknown subset for Pareto-Set subsampling: "{subset}"')
                    score += best_score ** 2 if subset != "EPS" else max(score, best_score)
                scores += [score]
        return tentatives[scores.index(min(scores))]  # type: ignore
