# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Any, Callable, List, Dict
import numpy as np
from nevergrad.common.typetools import ArrayLike
from .pyhv import _HyperVolume


ArgsKwargs = Tuple[Tuple[Any, ...], Dict[str, Any]]


class MultiobjectiveFunction:
    """Given several functions, and threshold on their values (above which solutions are pointless),
    this function returns a single-objective function, correctly instrumented, the minimization of which
    yields a solution to the original multiobjective problem.

    multi_objective_function: objective functions, to be minimized, of the original multiobjective problem.
    upper_bounds: upper_bounds[i] is a threshold above which x is pointless if functions[i](x) > upper_bounds[i].

    Returns an objective function to be minimized (it is a single objective function).
    Warning: this function is not stationary.
    The minimum value obtained for this objective function is -h,
    where h is the hypervolume of the Pareto front obtained, given upper_bounds as a reference point.
    """

    def __init__(self, multiobjective_function: Callable[..., ArrayLike], upper_bounds: ArrayLike) -> None:
        self.multiobjective_function = multiobjective_function
        self._upper_bounds = np.array(upper_bounds, copy=False)
        self._hypervolume: Any = _HyperVolume(self._upper_bounds)  # type: ignore
        self._points: List[Tuple[ArgsKwargs, np.ndarray]] = []
        self._best_volume = -float("Inf")

    def compute_aggregate_loss(self, losses: ArrayLike, *args: Any, **kwargs: Any) -> float:
        # We compute the hypervolume
        if (losses - self._upper_bounds > 0).any():
            return np.max(losses - self._upper_bounds)
        arr_losses = np.minimum(np.array(losses, copy=False), self._upper_bounds)
        new_volume: float = self._hypervolume.compute([y for _, y in self._points] + [arr_losses])
        if new_volume > self._best_volume:  # This point is good! Let us give him a great mono-fitness value.
            self._best_volume = new_volume
            # if tuple(x) in self.pointset:  # TODO: comparison is not quite possible, is it necessary?
            #    assert v == self.pointset[tuple(x)]  # We work noise-free...
            self._points.append(((args, kwargs), arr_losses))
            # self.pointset[tuple(x)] = v
            return -new_volume
        else:
            # Now we compute for each axis
            # First we prune.
            self.pareto_front
            distance_to_pareto = float("Inf")
            for _, stored_losses in self._points:
                print("we meet ", stored_losses)
                if (stored_losses <= arr_losses).all():
                    print("distance = ", min(arr_losses - stored_losses))
                    distance_to_pareto = min(distance_to_pareto, min(arr_losses - stored_losses))
                    print("now we have ", distance_to_pareto)
            assert distance_to_pareto >= 0
            return -new_volume + distance_to_pareto

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        # This part is stationary. It can be distributed.
        losses = self.multiobjective_function(*args, **kwargs)
        # The following is not. It should be called locally.
        return self.compute_aggregate_loss(losses, *args, **kwargs)

    @property
    def pareto_front(self) -> List[ArgsKwargs]:
        """Pareto front, as a list of args and kwargs (tuple of a tuple and a dict)
        for the function
        """
        new_points: List[Tuple[ArgsKwargs, np.ndarray]] = []
        for argskwargs, losses in self._points:
            should_be_added = True
            for _, other_losses in self._points:
                if (other_losses <= losses).all() and (other_losses < losses).any():
                    should_be_added = False
                    break
            if should_be_added:
                new_points.append((argskwargs, losses))
        self._points = new_points
        return [p[0] for p in self._points]
