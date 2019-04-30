# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Any
import numpy as np
from ..common.typetools import ArrayLike
from . import utils


def doerr_discrete_mutation(parent: ArrayLike) -> ArrayLike:
    """Mutation as in the fast 1+1-ES, Doerr et al. The exponent is 1.5.
    """
    dimension = len(parent)
    if dimension < 5:
        return discrete_mutation(parent)
    return doubledoerr_discrete_mutation(parent, max_ratio=.5)


def doubledoerr_discrete_mutation(parent: ArrayLike, max_ratio: float = 1.) -> ArrayLike:
    """Doerr's recommendation above can mutate up to half variables
    in average.
    In our high-arity context, we might need more than that.

    Parameters
    ----------
    parent: array-like
        the point to mutate
    max_ratio: float (between 0 and 1)
        the maximum mutation ratio (careful: this is not an exact ratio)
    """
    assert 0 <= max_ratio <= 1
    dimension = len(parent)
    max_mutations = int(max_ratio * dimension)
    p = 1. / np.arange(1, max_mutations)**1.5
    p /= np.sum(p)
    u = np.random.choice(np.arange(1, max_mutations), p=p)
    return portfolio_discrete_mutation(parent, u=u)


def portfolio_discrete_mutation(parent: ArrayLike, u: Optional[int] = None) -> ArrayLike:
    """Mutation discussed in
    https://arxiv.org/pdf/1606.05551v1.pdf
    We mutate a randomly drawn number of variables in average.
    """
    dimension = len(parent)
    if u is None:
        u = 1 if dimension == 1 else int(np.random.randint(1, dimension))
    boolean_vector = [True for _ in parent]
    while all(boolean_vector) and dimension != 1:
        boolean_vector = [np.random.rand() > (float(u) / dimension) for _ in parent]
    return [s if b else np.random.normal(0., 1.) for (b, s) in zip(boolean_vector, parent)]


def discrete_mutation(parent: ArrayLike) -> ArrayLike:
    dimension = len(parent)
    boolean_vector = [True for _ in parent]
    while all(boolean_vector):
        boolean_vector = [np.random.rand() > (1. / dimension) for _ in parent]
    return [s if b else np.random.normal(0., 1.) for (b, s) in zip(boolean_vector, parent)]


def crossover(parent: ArrayLike, donor: ArrayLike) -> ArrayLike:
    mix = [np.random.choice([d, p]) for (p, d) in zip(parent, donor)]
    return discrete_mutation(mix)


def get_roulette(archive: utils.Archive[utils.Value], num: Optional[int] = None) -> Any:
    """Apply a roulette tournament selection.
    """
    if num is None:
        num = int(.999 + np.sqrt(len(archive)))
    # the following sort makes the line deterministic, and function seedable, at the cost of complexity!
    my_keys = sorted(archive.bytesdict.keys())
    my_keys_indices = np.random.choice(len(my_keys), size=min(num, len(my_keys)), replace=False)
    my_keys = [my_keys[i] for i in my_keys_indices]
    # best pessimistic value in a random set of keys
    return np.frombuffer(min(my_keys, key=lambda x: archive.bytesdict[x].pessimistic_confidence_bound))
