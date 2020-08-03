# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import collections
import numpy as np
import pandas as pd
import nevergrad.common.typing as tp
from nevergrad.common import testing


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
    def read_csv(cls, path: tp.PathLike) -> "Selector":
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
