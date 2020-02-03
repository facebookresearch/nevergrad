# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import operator
import typing as tp
from uuid import uuid4
from collections import OrderedDict
import numpy as np
from nevergrad.common.tools import OrderedSet
from nevergrad.common.typetools import ArrayLike
from nevergrad.parametrization import parameter as p


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


def _get_nash(optimizer: tp.Any) -> tp.List[tp.Tuple[tp.Tuple[float, ...], int]]:
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


def sample_nash(optimizer: tp.Any) -> tp.Tuple[float, ...]:   # Somehow like fictitious play.
    nash = _get_nash(optimizer)
    if len(nash) == 1:
        return nash[0][0]
    prob = [float(n[1]) for n in nash]
    prob = [p_ / sum(prob) for p_ in prob]
    index: int = np.random.choice(np.arange(len(prob)), p=prob)
    return nash[index][0]


class DelayedJob:
    """Future-like object which delays computation
    """

    def __init__(self, func: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._result: tp.Optional[tp.Any] = None
        self._computed = False

    def done(self) -> bool:
        return True

    def result(self) -> tp.Any:
        if not self._computed:
            self._result = self.func(*self.args, **self.kwargs)
            self._computed = True
        return self._result


class SequentialExecutor:
    """Executor which run sequentially and locally
    (just calls the function and returns a FinishedJob)
    """

    def submit(self, fn: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> DelayedJob:
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


Y = tp.TypeVar("Y")


class Archive(tp.Generic[Y]):
    """A dict-like object with numpy arrays as keys.
    The underlying `bytesdict` dict stores the arrays as bytes since arrays are not hashable.
    Keys can be converted back with np.frombuffer(key)
    """

    def __init__(self) -> None:
        self.bytesdict: tp.Dict[bytes, Y] = {}

    def __setitem__(self, x: ArrayLike, value: Y) -> None:
        self.bytesdict[_tobytes(x)] = value

    def __getitem__(self, x: ArrayLike) -> Y:
        return self.bytesdict[_tobytes(x)]

    def __contains__(self, x: ArrayLike) -> bool:
        return _tobytes(x) in self.bytesdict

    def get(self, x: ArrayLike, default: tp.Optional[Y] = None) -> tp.Optional[Y]:
        return self.bytesdict.get(_tobytes(x), default)

    def __len__(self) -> int:
        return len(self.bytesdict)

    def values(self) -> tp.ValuesView[Y]:
        return self.bytesdict.values()

    def keys(self) -> None:
        raise RuntimeError(_ERROR_STR)

    def items(self) -> None:
        raise RuntimeError(_ERROR_STR)

    def items_as_array(self) -> tp.Iterator[tp.Tuple[np.ndarray, Y]]:
        raise RuntimeError("For consistency, items_as_array is renamed to items_as_arrays")

    def items_as_arrays(self) -> tp.Iterator[tp.Tuple[np.ndarray, Y]]:
        """Functions that iterates on key-values but transforms keys
        to np.ndarray. This is to simplify interactions, but should not
        be used in an algorithm since the conversion can be inefficient.
        Prefer using self.bytesdict.items() directly, and convert the bytes
        to np.ndarray using np.frombuffer(b)
        """
        return ((np.frombuffer(b), v) for b, v in self.bytesdict.items())

    def keys_as_array(self) -> tp.Iterator[np.ndarray]:
        raise RuntimeError("For consistency, keys_as_array is renamed to keys_as_arrays")

    def keys_as_arrays(self) -> tp.Iterator[np.ndarray]:
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
        quantiles: tp.Dict[str, float] = {}
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


class Individual:

    def __init__(self, x: ArrayLike) -> None:
        self.x = np.array(x, copy=False)
        self.uid = uuid4().hex
        self.value: tp.Optional[float] = None
        self._parameters = np.array([])
        self._active = True

    def __repr__(self) -> str:
        return f"Indiv<{self.x}, {self.value}>"


X = tp.TypeVar('X', bound=tp.Union[Individual, p.Parameter])


class Population(tp.Generic[X]):
    """Handle a population
    This could have a nicer interface... but it is already good enough
    Note: favor the use of UidQueue instead
    """

    def __init__(self, particles: tp.Iterable[X]) -> None:
        self._particles = OrderedDict({p.uid: p for p in particles})  # dont modify manually (needs updated uid to index)
        self._queue = tp.Deque[str]()
        self._uids: tp.List[str] = []
        self.extend(self._particles.values())

    @property
    def uids(self) -> tp.List[str]:
        """Don't modify manually
        """
        return self._uids

    def __repr__(self) -> str:
        particles = [p for p in self._particles.values()]
        return f"Population({particles})"

    def __getitem__(self, uid: str) -> X:
        parti = self._particles[uid]
        if parti.uid != uid:
            raise RuntimeError("Something went horribly wrong in the Population structure")
        return parti

    def __iter__(self) -> tp.Iterator[X]:
        return iter(self._particles.values())

    def extend(self, particles: tp.Iterable[X]) -> None:
        """Adds new particles
        The new particles are queued left (first out of queue)
        """
        particles = list(particles)
        self._uids.extend(p.uid for p in particles)
        self._particles.update({p.uid: p for p in particles})  # dont modify manually (needs updated uid to index)
        self._queue.extendleft(p.uid for p in reversed(particles))

    def __len__(self) -> int:
        return len(self._particles)

    def is_queue_empty(self) -> bool:
        return not self._queue

    def get_queued(self, remove: bool = False) -> X:
        if not self._queue:
            raise RuntimeError("Queue is empty, you tried to ask more than population size")
        uid = self._queue[0]  # pylint: disable=unsubscriptable-object
        if remove:
            self._queue.popleft()
        return self._particles[uid]

    def set_queued(self, particle: X) -> None:
        if particle.uid not in self._particles:
            raise ValueError("Individual is not part of the population")
        self._queue.append(particle.uid)

    def replace(self, oldie: X, newbie: X) -> None:
        """Replaces an old particle by a new particle.
        The new particle is queue left (first out of queue)
        """
        if oldie.uid not in self._particles:
            raise ValueError("Individual is not part of the population")
        if newbie.uid in self._particles:
            raise ValueError("Individual is already in the population")
        del self._particles[oldie.uid]
        self._particles[newbie.uid] = newbie
        self._uids = [newbie.uid if u == oldie.uid else u for u in self._uids]
        # update queue
        try:
            self._queue.remove(oldie.uid)
        except ValueError:
            pass
        self._queue.appendleft(newbie.uid)


class UidQueue:
    """Queue of uids to handle a population. This keeps track of:
    - told uids
    - asked uids
    When telling, it removes from the asked queue and adds to the told queue
    When asking, it takes from the told queue if not empty, else from the older
    asked, and then adds to the asked queue.
    """

    def __init__(self) -> None:
        self.told = tp.Deque[str]()
        self.asked: OrderedSet[str] = OrderedSet()

    def clear(self) -> None:
        """Removes all uids from the queues
        """
        self.told.clear()
        self.asked.clear()

    def ask(self) -> str:
        """Takes a uid from the told queue if not empty, else from the older asked,
        then adds it to the asked queue.
        """
        if self.told:
            uid = self.told.popleft()
        elif self.asked:
            uid = next(iter(self.asked))
        else:
            raise RuntimeError("Both asked and told queues are empty.")
        self.asked.add(uid)
        return uid

    def tell(self, uid: str) -> None:
        """Removes the uid from the asked queue and adds to the told queue
        """
        self.told.append(uid)
        if uid in self.asked:
            self.asked.discard(uid)

    def discard(self, uid: str) -> None:
        if uid in self.asked:
            self.asked.discard(uid)
        else:
            self.told.remove(uid)
