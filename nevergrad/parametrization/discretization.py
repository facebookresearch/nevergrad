# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Union, Any
import warnings
import numpy as np
import scipy.stats
from ..common.typetools import ArrayLike


# Nevergrad, in the most fundamental layer, uses continuous variables only.
# Discrete variables are handled in one of the following ways:
# - by a softmax transformation, a k-valued categorical variable is converted into k continuous variables.
# - by a discretization - as we often use Gaussian random values, we discretize according to quantiles of the normal
#   distribution.
def threshold_discretization(x: ArrayLike, arity: int = 2) -> List[int]:
    """Discretize by casting values from 0 to arity -1, assuming that x values
    follow a normal distribution.

    Parameters
    ----------
    x: list/array
       values to discretize
    arity: int
       the number of possible integer values (arity n will lead to values from 0 to n - 1)

    Note
    ----
    - nans are processed as negative infs (yields 0)
    """
    x = np.array(x, copy=True)
    if np.any(np.isnan(x)):
        warnings.warn("Encountered NaN values for discretization")
        x[np.isnan(x)] = -np.inf
    if arity == 2:  # special case, to have 0 yield 0
        return (np.array(x) > 0).astype(int).tolist()  # type: ignore
    else:
        return np.clip(arity * scipy.stats.norm.cdf(x), 0, arity - 1).astype(int).tolist()  # type: ignore


# The function below is the opposite of the function above.
def inverse_threshold_discretization(indexes: List[int], arity: int = 2) -> np.ndarray:
    indexes_arr = np.array(indexes, copy=True)
    pdf_bin_size = 1 / arity
    # We take the center of each bin (in the pdf space)
    return scipy.stats.norm.ppf(indexes_arr * pdf_bin_size + (pdf_bin_size / 2))  # type: ignore


# The discretization is, by nature, not one to one.
# In the function below, we randomly draw one of the possible inverse values - this is therefore noisy.
def noisy_inverse_threshold_discretization(indexes: List[int], arity: int = 2, gen: Any = None) -> np.ndarray:
    indexes_arr = np.array(indexes, copy=True)
    pdf_bin_size = 1 / arity
    # We take a random point in the bin.
    return scipy.stats.norm.ppf(indexes_arr * pdf_bin_size + gen.rand() * pdf_bin_size)  # type: ignore


def softmax_discretization(x: ArrayLike, arity: int = 2, random: Union[bool, np.random.RandomState] = True) -> List[int]:
    """Discretize a list of floats to a list of ints based on softmax probabilities.
    For arity n, a softmax is applied to the first n values, and the result
    serves as probability for the first output integer. The same process it
    applied to the other input values.

    Parameters
    ----------
    x: list/array
        the float values from a continuous space which need to be discretized
    arity: int
        the number of possible integer values (arity 2 will lead to values in {0, 1})
    random: bool or np.random.RandomState
        either a RandomState to pull values from, or True for pulling values on the default random state,
        or False to get a deterministic behavior

    Notes
    -----
    - if one or several inf values are present, only those are considered
    - in case of tie, the deterministic value is the first one (lowest) of the tie
    - nans and -infs are ignored, except if all are (then uniform random choice)
    """
    data = np.array(x, copy=True, dtype=float).reshape((-1, arity))
    if np.any(np.isnan(data)):
        warnings.warn("Encountered NaN values for discretization")
        data[np.isnan(data)] = -np.inf
    if random is False:
        output = np.argmax(data, axis=1).tolist()
        return output  # type: ignore
    if isinstance(random, bool):  # equivalent to "random is True"
        random = np.random  # default random number generator (creating a RandomState is slow)
    return [random.choice(arity, p=softmax_probas(d)) for d in data]


def softmax_probas(data: np.ndarray) -> np.ndarray:
    # TODO: test directly? (currently through softmax discretization)
    # TODO: move nan case here?
    maxv = np.max(data)
    if np.abs(maxv) == np.inf or np.isnan(maxv):
        maxv = 0
    data = np.exp(data - maxv)
    if any(x == np.inf for x in data):  # deal with infinite positives special case
        data = np.array([int(x == np.inf) for x in data])
    if not sum(data):
        data = np.ones(len(data))
    return data / np.sum(data)  # type: ignore


def inverse_softmax_discretization(index: int, arity: int) -> np.ndarray:
    # p is an arbitrary probability that the provided arg will be sampled with the returned point
    p = (1 / arity) * 1.5
    x: np.ndarray = np.zeros(arity)
    x[index] = np.log((p * (arity - 1)) / (1 - p))
    return x
