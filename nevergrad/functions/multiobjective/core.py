# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import nevergrad.common.typing as tp
from .hypervolume import HypervolumeIndicator


# pylint: disable=too-many-instance-attributes
class MultiobjectiveFunction:
    """Given several functions, and threshold on their values (above which solutions are pointless),
    this object can be used as a single-objective function, the minimization of which
    yields a solution to the original multiobjective problem.

    Parameters
    -----------
    multi_objective_function: callable
        objective functions, to be minimized, of the original multiobjective problem.
    upper_bounds: Tuple of float or np.ndarray
        upper_bounds[i] is a threshold above which x is pointless if function(x)[i] > upper_bounds[i].

    Notes
    -----
    - This function is not stationary!
    - The minimum value obtained for this objective function is -h,
      where h is the hypervolume of the Pareto front obtained, given upper_bounds as a reference point.
    - The callable keeps track of the pareto_front (see attribute paretor_front) and is therefor stateful.
      For this reason it cannot be distributed. A user can however call the multiobjective_function
      remotely, and aggregate locally. This is what happens in the "minimize" method of optimizers.
    """

    def __init__(self, multiobjective_function: tp.Callable[..., tp.ArrayLike], upper_bounds: tp.Optional[tp.ArrayLike] = None) -> None:
        self.multiobjective_function = multiobjective_function
        self._auto_bound = 0
        self._auto_upper_bounds = np.array([-float('inf')])
        self._auto_lower_bounds = np.array([float('inf')])
        if upper_bounds is None:
            self._auto_bound = 15
        else:
            self._upper_bounds = upper_bounds
            self._hypervolume: tp.Any = HypervolumeIndicator(self._upper_bounds)  # type: ignore
        self._points: tp.List[tp.Tuple[tp.ArgsKwargs, np.ndarray]] = []
        self._best_volume = -float("Inf")

    def compute_aggregate_loss(self, losses: tp.ArrayLike, *args: tp.Any, **kwargs: tp.Any) -> float:
        """Given parameters and the multiobjective loss, this computes the hypervolume
        and update the state of the function with new points if it belongs to the pareto front
        """
        losses = np.array(losses, copy=False)
        if self._auto_bound > 0:
            self._auto_upper_bounds = np.maximum(self._auto_upper_bounds, losses)  # type: ignore
            self._auto_lower_bounds = np.minimum(self._auto_lower_bounds, losses)  # type: ignore
            self._auto_bound -= 1
            if self._auto_bound == 0:
                self._upper_bounds = self._auto_upper_bounds + 0. * (self._auto_upper_bounds - self._auto_lower_bounds)
                self._hypervolume = HypervolumeIndicator(self._upper_bounds)  # type: ignore
            self._points.append(((args, kwargs), np.array(losses)))
            return 0.
        # We compute the hypervolume
        if (losses - self._upper_bounds > 0).any():
            return np.max(losses - self._upper_bounds)  # type: ignore
        arr_losses = np.minimum(np.array(losses, copy=False), self._upper_bounds)
        new_volume: float = self._hypervolume.compute([y for _, y in self._points] + [arr_losses])
        if new_volume > self._best_volume:  # This point is good! Let us give him a great mono-fitness value.
            self._best_volume = new_volume
            self._points.append(((args, kwargs), arr_losses))
            return -new_volume
        else:
            # Now we compute for each axis
            # First we prune.
            self._filter_pareto_front()
            distance_to_pareto = float("Inf")
            for _, stored_losses in self._points:
                if (stored_losses <= arr_losses).all():
                    distance_to_pareto = min(distance_to_pareto, min(arr_losses - stored_losses))
            assert distance_to_pareto >= 0
            return -new_volume + distance_to_pareto

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        # This part is stationary. It can be distributed.
        losses = self.multiobjective_function(*args, **kwargs)
        # The following is not. It should be called locally.
        return self.compute_aggregate_loss(losses, *args, **kwargs)

    def _filter_pareto_front(self) -> None:
        """filters the Pareto front, as a list of args and kwargs (tuple of a tuple and a dict).
        """
        new_points: tp.List[tp.Tuple[tp.ArgsKwargs, np.ndarray]] = []
        for argskwargs, losses in self._points:
            should_be_added = True
            for _, other_losses in self._points:
                if (other_losses <= losses).all() and (other_losses < losses).any():
                    should_be_added = False
                    break
            if should_be_added:
                new_points.append((argskwargs, losses))
        self._points = new_points

    def pareto_front(self, size: tp.Optional[int] = None, subset: str = "random") -> tp.List[tp.ArgsKwargs]:
        """Pareto front, as a list of args and kwargs (tuple of a tuple and a dict)

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "domain-covering", "hypervolume")

        Returns
        --------
        list
            the list of elements of the pareto front
        """
        self._filter_pareto_front()
        if size is None or size >= len(self._points):  # No limit: we return the full set.
            return [p[0] for p in self._points]
        if subset == "random":
            return random.sample([p[0] for p in self._points], size)
        possibilities: tp.List[tp.Any] = []
        scores: tp.List[float] = []
        for _ in range(30):
            possibilities += [random.sample(self._points, size)]
            if subset == "hypervolume":
                scores += [-self._hypervolume.compute([y for _, y in possibilities[-1]])]
            else:
                score: float = 0.
                for v, vloss in self._points:
                    best_score = float("inf")
                    for p, ploss in possibilities[-1]:
                        if subset == "loss-covering":
                            best_score = min(best_score, np.linalg.norm(ploss - vloss))
                        elif subset == "domain-covering":
                            best_score = min(best_score, np.linalg.norm(tuple(i - j for i, j in zip(p[0][0], v[0][0]))))
                        else:
                            raise ValueError(f'Unknown subset for Pareto-Set subsampling: "{subset}"')
                    score += best_score ** 2
                scores += [score]
        return [p[0] for p in possibilities[scores.index(min(scores))]]
