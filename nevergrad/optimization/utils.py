# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import operator
from typing import Tuple, Any, Callable, List
import numpy as np


class Value:
    """Estimation of a value based on one or multiple evaluations.
    This class provides easy access to:
    - count: how many times the point was evaluated
    - mean: the mean value.
    - square: the mean square value
    - variance: the variance

    It also provides access to optimistic and pessimistic bounds for the value.

    Parameter
    ---------
    y: float
        the first evaluation of the value
    """

    def __init__(self, y: float) -> None:
        self.count = 1
        self.mean = y
        self.square = y * y
        # TODO May be safer to use a default variance which depends on y for scale invariance?
        self.variance = 1.e6

    @property
    def optimistic_confidence_bound(self) -> float:
        return float(self.mean - .1 * np.sqrt((self.variance) / (1 + self.count)))

    @property
    def pessimistic_confidence_bound(self) -> float:
        return float(self.mean + .1 * np.sqrt((self.variance) / (1 + self.count)))

    def get_estimation(self, name: str) -> float:
        if name == "optimistic":
            return self.optimistic_confidence_bound
        elif name == "pessimistic":
            return self.pessimistic_confidence_bound
        else:
            raise NotImplementedError

    def add_evaluation(self, y: float) -> None:
        """Adds a new evaluation of the value

        Parameter
        ---------
        y: float
            the new evaluation
        """
        self.mean = (self.count * self.mean + y) / float(self.count + 1)
        self.square = (self.count * self.square + y * y) / float(self.count + 1)
        self.square = max(self.square, self.mean**2)
        self.count += 1
        factor: float = np.sqrt(float(self.count) / float(self.count - 1.))
        self.variance = factor * (self.square - self.mean**2)

    def __repr__(self) -> str:
        return "Value<mean: {}, count: {}>".format(self.mean, self.count)


class Point(Value):
    """Coordinates and estimation of a point in space.
    This class provides easy access to:
    - x: the coordinates of the point
    - count: how many times the point was evaluated
    - mean: the mean value.
    - square: the mean square value
    - variance: the variance

    It also provides access to optimistic and pessimistic bounds for the value.

    Parameters
    ----------
    x: array-like
        the coordinates
    value: Value
        the value estimation instance
    """

    def __init__(self, x: Tuple[float, ...], value: Value) -> None:
        assert isinstance(value, Value)
        super().__init__(value.mean)
        self.__dict__.update(value.__dict__)
        self.x = x

    def __repr__(self) -> str:
        return "Point<x: {}, mean: {}, count: {}>".format(self.x, self.mean, self.count)


def _get_nash(optimizer: Any) -> List[Tuple[Tuple[float, ...], int]]:
    """Returns an empirical distribution. limited using a threshold
    equal to max_num_trials^(1/4).
    """
    if not optimizer.archive:
        return [(optimizer.current_bests["pessimistic"].x, 1)]
    max_num_trial = max(p.count for p in optimizer.archive.values())
    sum_num_trial = sum(p.count for p in optimizer.archive.values())
    threshold = np.power(max_num_trial, .5)
    if threshold <= np.power(sum_num_trial, .25):
        return [(optimizer.provide_recommendation(), 1)]
    # make deterministic at the price of sort complexity
    return sorted(((k, p.count) for k, p in optimizer.archive.items() if p.count >= threshold),
                  key=operator.itemgetter(1))


def sample_nash(optimizer: Any) -> Tuple[float, ...]:   # Somehow like fictitious play.
    nash = _get_nash(optimizer)
    if len(nash) == 1:
        return nash[0][0]
    p = [float(n[1]) for n in nash]
    p = [p_ / sum(p) for p_ in p]
    index: int = np.random.choice(np.arange(len(p)), p=p)
    return nash[index][0]


class FinishedJob:
    """Future-like object with a pre-computed value
    """

    def __init__(self, result: Any) -> None:
        self._result = result

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        return self._result


class SequentialExecutor:
    """Executor which run sequentially and locally
    (just calls the function and returns a FinishedJob)
    """

    def submit(self, function: Callable, *args: Any, **kwargs: Any) -> FinishedJob:
        return FinishedJob(function(*args, **kwargs))
