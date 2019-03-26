# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import operator
from uuid import uuid4
from collections import OrderedDict, defaultdict
from typing import (Tuple, Any, Callable, List, Optional, Dict, ValuesView, Iterator,
                    TypeVar, Generic, Union, Deque, Iterable)
import numpy as np
from ..common.typetools import ArrayLike


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
        # Note: pruning below relies on the fact than only 3 modes exist. If a new mode is added, update pruning
        if name == "optimistic":
            return self.optimistic_confidence_bound
        elif name == "pessimistic":
            return self.pessimistic_confidence_bound
        elif name == "average":
            return self.mean
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

    def __init__(self, x: ArrayLike, value: Value) -> None:
        assert isinstance(value, Value)
        super().__init__(value.mean)
        self.__dict__.update(value.__dict__)
        assert not isinstance(x, (str, bytes))
        self.x = np.array(x, copy=True)  # copy to avoid interfering with algorithms
        self.x.flags.writeable = False  # make sure it is not modified!

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
    return sorted(((np.frombuffer(k), p.count) for k, p in optimizer.archive.bytesdict.items() if p.count >= threshold),
                  key=operator.itemgetter(1))


def sample_nash(optimizer: Any) -> Tuple[float, ...]:   # Somehow like fictitious play.
    nash = _get_nash(optimizer)
    if len(nash) == 1:
        return nash[0][0]
    p = [float(n[1]) for n in nash]
    p = [p_ / sum(p) for p_ in p]
    index: int = np.random.choice(np.arange(len(p)), p=p)
    return nash[index][0]


class DelayedJob:
    """Future-like object which delays computation
    """

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result: Optional[Any] = None
        self._computed = False

    def done(self) -> bool:
        return True

    def result(self) -> Any:
        if not self._computed:
            self._result = self.func(*self.args, **self.kwargs)
            self._computed = True
        return self._result


class SequentialExecutor:
    """Executor which run sequentially and locally
    (just calls the function and returns a FinishedJob)
    """

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> DelayedJob:
        return DelayedJob(fn, *args, **kwargs)


def _tobytes(x: ArrayLike) -> bytes:
    x = np.array(x, copy=False)  # for compatibility
    assert x.ndim == 1, f"Input shape: {x.shape}"
    assert x.dtype == np.float
    return x.tobytes()


_ERROR_STR = ("Generating numpy arrays from the bytes keys is inefficient, "
              "work on archive.bytesdict.<keys,items>() directly and convert with "
              "np.frombuffer if you can. You can also use archive.<keys,items>_as_arrays() "
              "but it is less efficient.")


Y = TypeVar("Y")


class Archive(Generic[Y]):
    """A dict-like object with numpy arrays as keys.
    The underlying `bytesdict` dict stores the arrays as bytes since arrays are not hashable.
    Keys can be converted back with np.frombuffer(key)
    """

    def __init__(self) -> None:
        self.bytesdict: Dict[bytes, Y] = {}

    def __setitem__(self, x: ArrayLike, value: Y) -> None:
        self.bytesdict[_tobytes(x)] = value

    def __getitem__(self, x: ArrayLike) -> Y:
        return self.bytesdict[_tobytes(x)]

    def __contains__(self, x: ArrayLike) -> bool:
        return _tobytes(x) in self.bytesdict

    def get(self, x: ArrayLike, default: Optional[Y] = None) -> Optional[Y]:
        return self.bytesdict.get(_tobytes(x), default)

    def __len__(self) -> int:
        return len(self.bytesdict)

    def values(self) -> ValuesView[Y]:
        return self.bytesdict.values()

    def keys(self) -> None:
        raise RuntimeError(_ERROR_STR)

    def items(self) -> None:
        raise RuntimeError(_ERROR_STR)

    def items_as_array(self) -> Iterator[Tuple[np.ndarray, Y]]:
        """Functions that iterates on key-values but transforms keys
        to np.ndarray. This is to simplify interactions, but should not
        be used in an algorithm since the conversion can be inefficient.
        Prefer using self.bytesdict.items() directly, and convert the bytes
        to np.ndarray using np.frombuffer(b)
        """
        return ((np.frombuffer(b), v) for b, v in self.bytesdict.items())

    def keys_as_array(self) -> Iterator[np.ndarray]:
        """Functions that iterates on keys but transforms them
        to np.ndarray. This is to simplify interactions, but should not
        be used in an algorithm since the conversion can be inefficient.
        Prefer using self.bytesdict.keys() directly, and convert the bytes
        to np.ndarray using np.frombuffer(b)
        """
        return (np.frombuffer(b) for b in self.bytesdict)

    def __repr__(self) -> str:
        return f"Archive with bytesdict: {self.bytesdict!r}"

    def __str__(self) -> str:
        return f"Archive with bytesdict: {self.bytesdict}"

    def __iter__(self) -> None:
        raise RuntimeError(_ERROR_STR)


class Pruning:
    """Callable for pruning archives in the optimizer class.
    See Optimizer.pruning attribute, called at each "tell".

    Parameters
    ----------
    min_len: int
        minimum length of the pruned archive.
    max_len: int
        length at which pruning is activated (maximum allowed length for the archive).

    Note
    ----
    For each of the 3 criteria (optimistic, pessimistic and average), the min_len best (lowest)
    points will be kept, which can lead to at most 3 * min_len points.
    """

    def __init__(self, min_len: int, max_len: int):
        self.min_len = min_len
        self.max_len = max_len

    def __call__(self, archive: Archive[Value]) -> Archive[Value]:
        if len(archive) < self.max_len:
            return archive
        warnings.warn("Pruning archive to save memory")
        quantiles: Dict[str, float] = {}
        threshold = float(self.min_len) / len(archive)
        names = ["optimistic", "pessimistic", "average"]
        for name in names:
            quantiles[name] = np.quantile([v.get_estimation(name) for v in archive.values()], threshold)
        new_archive = Archive[Value]()
        new_archive.bytesdict = {b: v for b, v in archive.bytesdict.items() if any(v.get_estimation(n) <= quantiles[n] for n in names)}
        return new_archive

    @classmethod
    def sensible_default(cls, num_workers: int, dimension: int) -> 'Pruning':
        """ Very conservative pruning
        - keep at least min_len 3 times num_workers
        - keep at most 30 times min_len or up to 1GB of array memory (whatever is biggest)

        Parameters
        ----------
        num_workers: int
            number of evaluations which will be run in parallel at once
        dimension: int
            dimension of the optimization space
        """
        # safer to keep at least 3 time the workers
        min_len = 3 * num_workers
        max_len = 10 * 3 * min_len  # len after pruning can be up to 3 min_len, amortize with an order of magnitude
        max_len_1gb = 1024**3 // (dimension * 8)
        return cls(min_len, max(max_len, max_len_1gb))


class Particle:

    def __init__(self) -> None:
        self.uuid = uuid4().hex
        self._waiting_for_removal = False


X = TypeVar('X', bound=Particle)


class Population(Generic[X]):
    """Handle a population
    This could have a nicer interface... but it is already good enough
    """

    def __init__(self, particles: Iterable[X]) -> None:
        self._particles = OrderedDict({p.uuid: p for p in particles})  # dont modify manually (needs updated uuid to index)
        self._link: Dict[Union[str, bytes, int], List[str]] = defaultdict(list)  # several particles can be linked to a same point
        self._queue = Deque[str]()
        self._uuids: List[str] = []
        self.extend(self._particles.values())

    @property
    def uuids(self) -> List[str]:
        """Don't modify manually
        """
        return self._uuids

    def __getitem__(self, uuid: str) -> X:
        parti = self._particles[uuid]
        if parti.uuid != uuid:
            raise RuntimeError("Something went horribly wrong in the Population structure")
        return parti

    def __iter__(self) -> Iterator[X]:
        return iter(self._particles.values())

    def extend(self, particles: Iterable[X]) -> None:
        """Adds new particles
        The new particles are queued left (first out of queue)
        """
        particles = list(particles)
        self._uuids.extend(p.uuid for p in particles)
        self._particles.update({p.uuid: p for p in particles})  # dont modify manually (needs updated uuid to index)
        self._queue.extendleft(p.uuid for p in reversed(particles))

    def __len__(self) -> int:
        return len(self._particles)

    def get_linked(self, key: Union[str, bytes, int]) -> X:
        uuids = self._link[key]
        if not uuids:
            raise KeyError("No link available")
        return self._particles[uuids[0]]

    def set_linked(self, key: Union[str, bytes, int], particle: X) -> None:
        if particle.uuid not in self._particles:
            raise ValueError("Particle is not part of the population")
        self._link[key].append(particle.uuid)

    def del_link(self, key: Union[str, bytes, int], particle: X) -> None:
        self._link[key].remove(particle.uuid)
        if not self._link[key]:
            del self._link[key]

    def is_queue_empty(self) -> bool:
        return not self._queue

    def get_queued(self, remove: bool = False) -> X:
        if not self._queue:
            raise RuntimeError("Queue is empty, you tried to ask more than population size")
        uuid = self._queue[0]  # pylint: disable=unsubscriptable-object
        if remove:
            self._queue.popleft()
        return self._particles[uuid]

    def set_queued(self, particle: X) -> None:
        if particle.uuid not in self._particles:
            raise ValueError("Particle is not part of the population")
        self._queue.append(particle.uuid)

    def replace(self, oldie: X, newbie: X) -> Optional[Union[str, bytes, int]]:
        """Replaces an old particle by a new particle.
        The new particle is queue left (first out of queue)
        If the old particle was linked, the key will be returned
        """
        if oldie.uuid not in self._particles:
            raise ValueError("Particle is not part of the population")
        if newbie.uuid in self._particles:
            raise ValueError("Particle is already in the population")
        del self._particles[oldie.uuid]
        self._particles[newbie.uuid] = newbie
        self._uuids = [newbie.uuid if u == oldie.uuid else u for u in self._uuids]
        # update queue
        try:
            self._queue.remove(oldie.uuid)
        except ValueError:
            pass
        self._queue.appendleft(newbie.uuid)
        # update dict
        links = [key for key, uuids in self._link.items() if oldie.uuid in uuids]
        assert len(links) <= 1
        if not links:
            return None
        else:
            del self._link[links[0]]
            return links[0]
