# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import scipy.stats
import nevergrad.common.typing as tp


# Nevergrad, in the most fundamental layer, uses continuous variables only.
# Discrete variables are handled in one of the following ways:
# - by a softmax transformation, a k-valued categorical variable is converted into k continuous variables.
# - by a discretization - as we often use Gaussian random values, we discretize according to quantiles of the normal
#   distribution.
def threshold_discretization(x: tp.ArrayLike, arity: int = 2) -> tp.List[int]:
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
def inverse_threshold_discretization(indexes: tp.List[int], arity: int = 2) -> np.ndarray:
    indexes_arr = np.array(indexes, copy=True)
    assert not np.any(np.isnan(indexes_arr))
    pdf_bin_size = 1 / arity
    # We take the center of each bin (in the pdf space)
    x = scipy.stats.norm.ppf(indexes_arr * pdf_bin_size + (pdf_bin_size / 2))  # type: ignore
    nan_indices = np.where(np.isnan(x))
    x[nan_indices] = np.sign(indexes_arr[nan_indices] - (arity / 2.0)) * np.finfo(np.dtype("float")).max
    return x


# The discretization is, by nature, not one to one.
# In the function below, we randomly draw one of the possible inverse values - this is therefore noisy.
def noisy_inverse_threshold_discretization(
    indexes: tp.List[int], arity: int = 2, gen: tp.Any = None
) -> np.ndarray:
    indexes_arr = np.array(indexes, copy=True)
    pdf_bin_size = 1 / arity
    # We take a random point in the bin.
    return scipy.stats.norm.ppf(indexes_arr * pdf_bin_size + gen.rand() * pdf_bin_size)  # type: ignore


def weight_for_reset(arity: int) -> float:
    """p is an arbitrary probability that the provided arg will be sampled with the returned point"""
    p = (1 / arity) * 1.5
    w = float(np.log((p * (arity - 1)) / (1 - p)))
    return w


class Encoder:
    """Handles softmax weights which need to be turned into probabilities and sampled
    This class is expected to evolve to be more usable and include more features (like
    conversion from probabilities to weights?)
    It will replace most of the code above if possible

    Parameters
    ----------
    weights: array
        the weights of size samples x options, that will be turned to probabilities
        using softmax.
    rng: RandomState
        random number generator for sampling following the probabilities

    Notes
    -----
    - if one or several inf values are present in a row, only those are considered
    - in case of tie, the deterministic value is the first one (lowest) of the tie
    - nans and -infs are ignored, except if all are (then uniform random choice)
    """

    def __init__(self, weights: np.ndarray, rng: np.random.RandomState) -> None:
        self.weights = np.array(weights, copy=True, dtype=float)
        self.weights[np.isnan(self.weights)] = -np.inf  # 0 proba for nan values
        self._rng = rng

    def probabilities(self) -> np.ndarray:
        """Creates the probability matrix from the weights"""
        axis = 1
        maxv = np.max(self.weights, axis=1, keepdims=True)
        hasposinf = np.isposinf(maxv)
        maxv[np.isinf(maxv)] = 0  # avoid indeterminations
        exp: np.ndarray = np.exp(self.weights - maxv)
        # deal with infinite positives special case
        # by ignoring (0 proba) non-infinte on same row
        if np.any(hasposinf):
            is_inf = np.isposinf(self.weights)
            is_ignored = np.logical_and(np.logical_not(is_inf), hasposinf)
            exp[is_inf] = 1
            exp[is_ignored] = 0
        # random choice if sums to 0
        sums0 = np.sum(exp, axis=axis) == 0
        exp[sums0, :] = 1
        exp /= np.sum(exp, axis=axis, keepdims=True)  # normalize
        return exp

    def encode(self, deterministic: bool = False) -> np.ndarray:
        """Sample an index from each row depending on the provided probabilities.

        Parameters
        ----------
        deterministic: bool
            set to True for sampling deterministically the more likely option
            (largest probability)
        """
        axis = 1
        if deterministic:
            return np.argmax(self.weights, axis=1)  # type: ignore
        cumprob = np.cumsum(self.probabilities(), axis=axis)
        rand = self._rng.rand(cumprob.shape[0], 1)
        return np.argmin(cumprob < rand, axis=axis)  # type: ignore
