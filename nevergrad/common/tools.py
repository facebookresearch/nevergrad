# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import inspect
import warnings
import itertools
import collections
import typing as tp
import numpy as np
import pandas as pd
from .typetools import PathLike
from . import testing


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
    """roundrobin('ABC', 'D', 'EF') --> A D E B F C
    """
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


class Selector(pd.DataFrame):  # type: ignore
    """Pandas dataframe class with a simplified selection function
    """

    @property
    def _constructor_expanddim(self) -> tp.Type["Selector"]:
        return Selector

    @property
    def _constructor(self) -> tp.Type["Selector"]:
        return Selector

    # pylint: disable=arguments-differ
    def select(self, **kwargs: tp.Union[str, tp.Sequence[str], tp.Callable[[tp.Any], bool]]) -> "Selector":
        """Select rows based on a value, a sequence of values or a discriminating function

        Parameters
        ----------
        kwargs: str, list or callable
            selects values in the column provided as keyword, based on str matching, or
            presence in the list, or callable returning non-False on the values

        Example
        -------
        df.select(column1=["a", "b"])
        will return a new Selector with rows having either "a" or "b" as value in column1
        """
        df = self
        for name, criterion in kwargs.items():
            if isinstance(criterion, collections.abc.Iterable) and not isinstance(criterion, str):
                selected = df.loc[:, name].isin(criterion)
            elif callable(criterion):
                selected = [bool(criterion(x)) for x in df.loc[:, name]]
            else:
                selected = df.loc[:, name].isin([criterion])
            df = df.loc[selected, :]
        return Selector(df)

    def select_and_drop(self, **kwargs: tp.Union[str, tp.Sequence[str], tp.Callable[[tp.Any], bool]]) -> "Selector":
        """Same as select, but drops the columns used for selection
        """
        df = self.select(**kwargs)
        columns = [x for x in df.columns if x not in kwargs]
        return Selector(df.loc[:, columns])

    def unique(self, column_s: tp.Union[str, tp.Sequence[str]]) -> tp.Union[tp.Tuple[tp.Any, ...], tp.Set[tp.Tuple[tp.Any, ...]]]:
        """Returns the set of unique values or set of values for a column or columns

        Parameter
        ---------
        column_s: str or tp.Sequence[str]
            a column name, or list of column names

        Returns
        -------
        set
           a set of values if the input was a column name, or a set of tuple of values
           if the name was a list of columns
        """
        if isinstance(column_s, str):
            return set(self.loc[:, column_s])  # equivalent to df.<name>.unique()
        elif isinstance(column_s, (list, tuple)):
            testing.assert_set_equal(set(column_s) - set(self.columns), {}, err_msg="Unknown column(s)")
            df = self.loc[:, column_s]
            assert not df.isnull().values.any(), "Cannot work with NaN values"
            return set(tuple(row) for row in df.itertuples(index=False))
        else:
            raise NotImplementedError("Only strings, lists and tuples are allowed")

    @classmethod
    def read_csv(cls, path: PathLike) -> "Selector":
        return cls(pd.read_csv(str(path)))

    def assert_equivalent(self, other: pd.DataFrame, err_msg: str = "") -> None:
        """Asserts that two selectors are equal, up to row and column permutations

        Note
        ----
        Use sparsely, since it is quite slow to test
        """
        testing.assert_set_equal(other.columns, self.columns, f"Different columns\n{err_msg}")
        np.testing.assert_equal(len(other), len(self), "Different number of rows\n{err_msg}")
        other_df = other.loc[:, self.columns]
        df_rows: tp.List[tp.List[tp.Tuple[tp.Any, ...]]] = [[], []]
        for k, df in enumerate([self, other_df]):
            for row in df.itertuples(index=False):
                df_rows[k].append(tuple(row))
            df_rows[k].sort()
        for row1, row2 in zip(*df_rows):
            np.testing.assert_array_equal(row1, row2, err_msg=err_msg)


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
            value = np.mean(self._queue)
        return float(np.clip(value / self._num_waits, self._min, self._max))

    def sleep(self) -> None:
        time.sleep(self._get_advised_sleep_duration())


X = tp.TypeVar("X", bound=tp.Hashable)


class OrderedSet(tp.MutableSet[X]):
    """Set of elements retaining the insertion order
    All new elements are appended to the end of the set.
    """

    def __init__(self, keys: tp.Optional[tp.Iterable[X]] = None) -> None:
        self._data: 'collections.OrderedDict[X, int]' = collections.OrderedDict()
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


def different_from_defaults(instance: tp.Any, check_mismatches: bool = False) -> tp.Dict[str, tp.Any]:
    """Checks which attributes are different from defaults arguments

    Parameters
    ----------
    instance: object
        the object to change
    check_mismatches: bool
        checks that the attributes match the parameters

    Note
    ----
    This is convenient for short repr of data structures
    """
    defaults = {
        x: y.default for x, y in inspect.signature(instance.__class__.__init__).parameters.items() if x not in ["self", "__class__"]
    }
    if check_mismatches:
        diff = set(defaults.keys()).symmetric_difference(instance.__dict__.keys())
        if diff:  # this is to help during development
            raise RuntimeError(f"Mismatch between attributes and arguments of {instance}: {diff}")
    else:
        defaults = {x: y for x, y in defaults.items() if x in instance.__dict__}
    # only print non defaults
    return {x: instance.__dict__[x] for x, y in defaults.items() if y != instance.__dict__[x] and not x.startswith("_")}
