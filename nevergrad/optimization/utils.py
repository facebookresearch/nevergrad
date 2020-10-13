# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import operator
import warnings
import numpy as np
from nevergrad.parametrization import parameter as p
import nevergrad.common.typing as tp
from nevergrad.common.tools import OrderedSet


class MultiValue:
    """Estimation of a value based on one or multiple evaluations.
    This class provides easy access to:
    - count: how many times the point was evaluated
    - mean: the mean value.
    - square: the mean square value
    - variance: the variance
    - parameter: the corresponding Parameter


    It also provides access to optimistic and pessimistic bounds for the value.

    Parameter
    ---------
    parameter: Parameter
        the parameter for one of the evaluations
    y: float
        the first evaluation of the value
    """

    def __init__(self, parameter: p.Parameter, y: float, *, reference: p.Parameter) -> None:
        self.count = 1
        self.mean = y
        self.square = y * y
        # TODO May be safer to use a default variance which depends on y for scale invariance?
        self.variance = 1.e6
        parameter.freeze()
        self.parameter = parameter
        self._ref = reference

    @property
    def x(self) -> np.ndarray:  # for compatibility
        return self.parameter.get_standardized_data(reference=self._ref)

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
        factor = math.sqrt(float(self.count) / float(self.count - 1.))
        self.variance = factor * (self.square - self.mean**2)

    def as_array(self, reference: p.Parameter) -> np.ndarray:
        return self.parameter.get_standardized_data(reference=reference)

    def __repr__(self) -> str:
        return f"MultiValue<mean: {self.mean}, count: {self.count}, parameter: {self.parameter}>"


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


def _tobytes(x: tp.ArrayLike) -> bytes:
    x = np.array(x, copy=False)  # for compatibility
    assert x.ndim == 1, f"Input shape: {x.shape}"
    assert x.dtype == np.float, f"Incorrect type {x.dtype} is not float"
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

    def __setitem__(self, x: tp.ArrayLike, value: Y) -> None:
        self.bytesdict[_tobytes(x)] = value

    def __getitem__(self, x: tp.ArrayLike) -> Y:
        return self.bytesdict[_tobytes(x)]

    def __contains__(self, x: tp.ArrayLike) -> bool:
        return _tobytes(x) in self.bytesdict

    def get(self, x: tp.ArrayLike, default: tp.Optional[Y] = None) -> tp.Optional[Y]:
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

    def __call__(self, archive: Archive[MultiValue]) -> Archive[MultiValue]:
        if len(archive) < self.max_len:
            return archive
        quantiles: tp.Dict[str, float] = {}
        threshold = float(self.min_len) / len(archive)
        names = ["optimistic", "pessimistic", "average"]
        for name in names:
            quantiles[name] = np.quantile([v.get_estimation(name) for v in archive.values()], threshold, interpolation="lower")
        new_archive: Archive[MultiValue] = Archive()
        new_archive.bytesdict = {b: v for b, v in archive.bytesdict.items() if any(v.get_estimation(n) <= quantiles[n] for n in names)}
        return new_archive

    @classmethod
    def sensible_default(cls, num_workers: int, dimension: int) -> 'Pruning':
        """ Very conservative pruning
        - keep at least 100 elements, or 7 times num_workers, whatever is biggest
        - keep at least 3 x min_len, or up to 10 x min_len if it does not exceed 1gb of data

        Parameters
        ----------
        num_workers: int
            number of evaluations which will be run in parallel at once
        dimension: int
            dimension of the optimization space
        """
        # safer to keep at least 7 time the workers
        min_len = max(100, 7 * num_workers)
        max_len_1gb = 1024**3 // (dimension * 8 * 2)  # stored twice: as key and as Parameter
        max_len = max(3 * min_len, min(10 * min_len, max_len_1gb))
        return cls(min_len, max_len)


class UidQueue:
    """Queue of uids to handle a population. This keeps track of:
    - told uids
    - asked uids
    When telling, it removes from the asked queue and adds to the told queue
    When asking, it takes from the told queue if not empty, else from the older
    asked, and then adds to the asked queue.
    """

    def __init__(self) -> None:
        self.told = tp.Deque[str]()  # this seems to be picklable (this syntax does not always work)
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


class BoundScaler:
    """Hacky way to sample in the space defined by the parametrization.
    Given an vector of values between 0 and 1,
    the transform method samples in the bounds if provided,
    or using the provided function otherwise.
    This is used for samplers.
    Code of parametrization and/or this helper should definitely be
    updated to make it simpler and more robust

    It warns in
    """

    def __init__(self, reference: p.Parameter) -> None:
        self.reference = reference.spawn_child()
        self.reference.freeze()
        # initial check
        parameter = self.reference.spawn_child()
        parameter.set_standardized_data(np.linspace(-1, 1, self.reference.dimension))
        expected = parameter.get_standardized_data(reference=self.reference)
        self._ref_arrays = self.list_arrays(self.reference)
        arrays = self.list_arrays(parameter)
        check = np.concatenate([x.get_standardized_data(reference=y) for x, y in zip(arrays, self._ref_arrays)], axis=0)
        self.working = True
        if not np.allclose(check, expected):
            self.working = False
            self._warn()

    def _warn(self) -> None:
        warnings.warn(f"Failed to find bounds for {self.reference}, quasi-random optimizer may be inefficient.\n"
                      "Please open an issue on Nevergrad github")

    @classmethod
    def list_arrays(cls, parameter: p.Parameter) -> tp.List[p.Array]:
        """Computes a list of data (Array) parameters in the same order as in
        the standardized data space.
        """
        if isinstance(parameter, p.Array):
            return [parameter]
        elif isinstance(parameter, p.Constant):
            return []
        if not isinstance(parameter, p.Dict):
            raise RuntimeError(f"Unsupported parameter {parameter}")
        output: tp.List[p.Array] = []
        for _, subpar in sorted(parameter._content.items()):
            output += cls.list_arrays(subpar)
        return output

    def transform(self, x: tp.ArrayLike, unbounded_transform: tp.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Transform from [0, 1] to the space between bounds
        """
        y = np.array(x, copy=True)
        if not self.working:
            return unbounded_transform(y)
        try:
            out = self._transform(y, unbounded_transform)
        except Exception:  # pylint: disable=broad-except
            self._warn()
            out = unbounded_transform(y)
        return out

    def _transform(self, x: np.ndarray, unbounded_transform: tp.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        # modifies x in place
        start = 0
        for ref in self._ref_arrays:
            end = start + ref.dimension
            if any(b is None for b in ref.bounds) or not ref.full_range_sampling:
                x[start: end] = unbounded_transform(x[start: end])
            else:
                array = ref.spawn_child()
                bounds: tp.List[tp.Any] = list(ref.bounds)
                if array.exponent is not None:
                    bounds = [np.log(b) for b in bounds]
                value = bounds[0] + (bounds[1] - bounds[0]) * x[start:end].reshape(ref._value.shape)
                if array.exponent is not None:
                    value = np.exp(value)
                array._value = value
                x[start: end] = array.get_standardized_data(reference=ref)
            start = end
        return x
