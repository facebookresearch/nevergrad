# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import site
import glob
import ctypes
import time
import inspect
import warnings
import itertools
import contextlib
import collections
import typing as tp
import numpy as np


def pytorch_import_fix() -> None:
    """Hackfix needed before pytorch import ("dlopen: cannot load any more object with static TLS")
    See issue #305
    """
    try:
        for packages in site.getsitepackages():
            for lib in glob.glob(f"{packages}/torch/lib/libgomp*.so*"):
                ctypes.cdll.LoadLibrary(lib)
    except Exception:  # pylint: disable=broad-except
        pass


def pairwise(iterable: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Tuple[tp.Any, tp.Any]]:
    """Returns an iterator over sliding pairs of the input iterator
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    Note
    ----
    Nothing will be returned if length of iterator is strictly less
    than 2.
    """  # From itertools documentation
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable: tp.Iterable[tp.Any], n: int, fillvalue: tp.Any = None) -> tp.Iterator[tp.List[tp.Any]]:
    """Collect data into fixed-length chunks or blocks
    Copied from itertools recipe documentation
    Example: grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def roundrobin(*iterables: tp.Iterable[tp.Any]) -> tp.Iterator[tp.Any]:
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C"""
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_ in nexts:
                yield next_()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


class Sleeper:
    """Simple object for managing the waiting time of a job

    Parameters
    ----------
    min_sleep: float
        minimum sleep time
    max_sleep: float
        maximum sleep time
    averaging_size: int
        size for averaging the registered durations
    """

    def __init__(self, min_sleep: float = 1e-7, max_sleep: float = 1.0, averaging_size: int = 10) -> None:
        self._min = min_sleep
        self._max = max_sleep
        self._start: tp.Optional[float] = None
        self._queue: tp.Deque[float] = collections.deque(maxlen=averaging_size)
        self._num_waits = 10  # expect to waste around 10% of time

    def start_timer(self) -> None:
        if self._start is not None:
            warnings.warn("Ignoring since timer was already started.")
            return
        self._start = time.time()

    def stop_timer(self) -> None:
        if self._start is None:
            warnings.warn("Ignoring since timer was stopped before starting.")
            return
        self._queue.append(time.time() - self._start)
        self._start = None

    def _get_advised_sleep_duration(self) -> float:
        if not self._queue:
            if self._start is None:
                return self._min
            value = time.time() - self._start
        else:
            value = float(np.mean(self._queue))
        return float(np.clip(value / self._num_waits, self._min, self._max))

    def sleep(self) -> None:
        time.sleep(self._get_advised_sleep_duration())


X = tp.TypeVar("X", bound=tp.Hashable)


class OrderedSet(tp.MutableSet[X]):
    """Set of elements retaining the insertion order
    All new elements are appended to the end of the set.
    """

    def __init__(self, keys: tp.Optional[tp.Iterable[X]] = None) -> None:
        self._data: "collections.OrderedDict[X, int]" = collections.OrderedDict()
        self._global_index = 0  # keep track of insertion global index if need be
        if keys is not None:
            for key in keys:
                self.add(key)

    def add(self, key: X) -> None:
        self._data[key] = self._data.pop(key, self._global_index)
        self._global_index += 1

    def popright(self) -> X:
        key = next(reversed(self._data))
        self.discard(key)
        return key

    def discard(self, key: X) -> None:
        del self._data[key]

    def __contains__(self, key: tp.Any) -> bool:
        return key in self._data

    def __iter__(self) -> tp.Iterator[X]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def different_from_defaults(
    *,
    instance: tp.Any,
    instance_dict: tp.Optional[tp.Dict[str, tp.Any]] = None,
    check_mismatches: bool = False,
) -> tp.Dict[str, tp.Any]:
    """Checks which attributes are different from defaults arguments

    Parameters
    ----------
    instance: object
        the object to change
    instance_dict: dict
        the dict corresponding to the instance, if not provided it's self.__dict__
    check_mismatches: bool
        checks that the attributes match the parameters

    Note
    ----
    This is convenient for short repr of data structures
    """
    defaults = {
        x: y.default
        for x, y in inspect.signature(instance.__class__.__init__).parameters.items()
        if x not in ["self", "__class__"]
    }
    if instance_dict is None:
        instance_dict = instance.__dict__
    if check_mismatches:
        add = set(defaults.keys()) - set(instance_dict.keys())
        miss = set(instance_dict.keys()) - set(defaults.keys())
        if add or miss:  # this is to help during development
            raise RuntimeError(
                f"Mismatch between attributes and arguments of {instance.__class__}:\n"
                f"- additional: {add}\n- missing: {miss}"
            )
    else:
        defaults = {x: y for x, y in defaults.items() if x in instance.__dict__}
    # only print non defaults
    return {
        x: instance_dict[x] for x, y in defaults.items() if y != instance_dict[x] and not x.startswith("_")
    }


@contextlib.contextmanager
def set_env(**environ: tp.Any) -> tp.Generator[None, None, None]:
    """Temporarily changes environment variables."""
    old_environ = {x: os.environ.get(x, None) for x in environ}
    for x in environ:
        if x != x.upper():
            raise ValueError(f"Only capitalized environment variable are allowed, but got {x!r}")
    os.environ.update({x: str(y) for x, y in environ.items()})
    try:
        yield
    finally:
        for k, val in old_environ.items():
            os.environ.pop(k)
            if val is not None:
                os.environ[k] = val


def flatten(obj: tp.Any) -> tp.Any:
    """Flatten a dict/list structure

    Example
    -------

    >>> flatten(["a", {"truc": [4, 5]}])
    >>> {"0": "a", "1.truc.0": 4, "1.truc.1": 5}
    """
    output: tp.Any = {}
    if isinstance(obj, (tuple, list)):
        iterator = enumerate(obj)
    elif isinstance(obj, dict):
        iterator = obj.items()  # type: ignore
    else:
        return obj
    for k, val in iterator:
        content = flatten(val)
        if isinstance(content, dict):
            output.update({f"{k}.{x}": y for x, y in content.items()})
        else:
            output[str(k)] = val
    return output
