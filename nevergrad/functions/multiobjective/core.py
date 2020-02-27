# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Any, Callable, List, Dict, Optional
import numpy as np
import random
from nevergrad.common.typetools import ArrayLike
from .hypervolume import HypervolumeIndicator

ArgsKwargs = Tuple[Tuple[Any, ...], Dict[str, Any]]


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

    def __init__(self, multiobjective_function: Callable[..., ArrayLike], upper_bounds: Optional[ArrayLike] = None) -> None:
        self.multiobjective_function = multiobjective_function
        self._bound_budget = 10
        if upper_bounds is None:
            self._upper_bounds = np.array([0.])
            self._auto_bound = self._bound_budget
        else:
            self._upper_bounds = np.array(upper_bounds, copy=False)
            self._auto_bound = 0
        self._hypervolume: Any = HypervolumeIndicator(self._upper_bounds)  # type: ignore
        self._points: List[Tuple[ArgsKwargs, np.ndarray]] = []
        self._best_volume = -float("Inf")

    def compute_aggregate_loss(self, losses: ArrayLike, *args: Any, **kwargs: Any) -> float:
        """Given parameters and the multiobjective loss, this computes the hypervolume
        and update the state of the function with new points if it belongs to the pareto front
        """
        arr_losses = np.array(losses, copy=True)  #np.minimum(np.array(losses, copy=False), self._upper_bounds)
        # We compute the hypervolume
        if self._auto_bound > 0:
            self._upper_bounds = arr_losses if self._auto_bound == self._bound_budget else np.maximum(self._upper_bounds, arr_losses)
            self._lower_bounds = arr_losses if self._auto_bound == self._bound_budget else np.minimum(self._lower_bounds, arr_losses)
            self._auto_bound -= 1
            #print("ub = ", self._upper_bounds, " lb = ", self._lower_bounds)
            if self._auto_bound <= 0:
                self._upper_bounds = self._upper_bounds + 2. * (self._upper_bounds - self._lower_bounds)
                self._filter_pareto_front()
            self._points.append(((args, kwargs), np.array(losses)))
            #print('whereas ', len(self._points))
            #print("*autobound...")
            self._best_volume = -float("Inf")
            self._hypervolume = HypervolumeIndicator(self._upper_bounds)  # type: ignore
            return float("0")
                
        if (losses - self._upper_bounds > 0).any() or self._auto_bound > 0:
            #print("*very bad...")
            return 1e7 + 1e7 * np.max(losses - self._upper_bounds)  # type: ignore
        #print("pareto set:")
        #print([y for _, y in self._points] + [arr_losses])
        new_volume: float = self._hypervolume.compute([y for _, y in self._points] + [arr_losses])
        #print('-->', new_volume)
        if new_volume > self._best_volume:  # This point is good! Let us give him a great mono-fitness value.
            #print(self._best_volume, "*great !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", new_volume)
            self._best_volume = new_volume
            # if tuple(x) in self.pointset:  # TODO: comparison is not quite possible, is it necessary?
            #    assert v == self.pointset[tuple(x)]  # We work noise-free...
            self._points.append(((args, kwargs), arr_losses))
            self._filter_pareto_front()
            return -new_volume
        else:
            #if new_volume == self._best_volume:
            #    #print("*tangent...", new_volume)
            #    return 0.
            # Now we compute for each axis
            # First we prune.
            if len(self._points) % 10 == 0:
                self._filter_pareto_front()
            distance_to_pareto = float("Inf")
            for _, stored_losses in self._points:
                if (stored_losses <= arr_losses).all():
                    distance_to_pareto = min(distance_to_pareto, min(arr_losses - stored_losses))
            assert distance_to_pareto >= 0
            #print("*bad")
            return -new_volume + distance_to_pareto

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        # This part is stationary. It can be distributed.
        losses = self.multiobjective_function(*args, **kwargs)
        # The following is not. It should be called locally.
        value = self.compute_aggregate_loss(losses, *args, **kwargs)
        return value

    def _filter_pareto_front(self):
        """filters the Pareto front, as a list of args and kwargs (tuple of a tuple and a dict).
        """
        new_points: List[Tuple[ArgsKwargs, np.ndarray]] = []
        for argskwargs, losses in self._points:
            should_be_added = True
            if (losses > self._upper_bounds).any():
                should_be_added = False
                break
            for _, other_losses in self._points:
                if (other_losses <= losses).all() and (other_losses < losses).any():
                    should_be_added = False
                    break
            if should_be_added:
                new_points.append((argskwargs, losses))
        #print(len(self._points), ' --whereas--> ', len(new_points))
        self._points = new_points


    def pareto_front(self, size: Optional[int] = None, subset: str = "random"): # -> List[ArgsKwargs], List[Any]:
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
            return [p[0] for p in self._points], [-3.141592]
        if subset == "random":
            return random.sample([p[0] for p in self._points], size), [3.141592]
        possibilities: List[Any] = []
        scores : List[float] = []
        localrandom = random.Random(0)
        for u in range(60):
            possibilities += [localrandom.sample(self._points, size)]
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
        return [p[0] for p in possibilities[scores.index(min(scores))]], scores
