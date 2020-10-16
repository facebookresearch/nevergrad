# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad.common.typing as tp
from . import utils


class Mutator:
    """Class defining mutations, and holding a random state used for random generation.
    """

    def __init__(self, random_state: np.random.RandomState) -> None:
        self.random_state = random_state

    def doerr_discrete_mutation(self, parent: tp.ArrayLike) -> tp.ArrayLike:
        """Mutation as in the fast 1+1-ES, Doerr et al. The exponent is 1.5.
        """
        dimension = len(parent)
        if dimension < 5:
            return self.discrete_mutation(parent)
        return self.doubledoerr_discrete_mutation(parent, max_ratio=.5)

    def doubledoerr_discrete_mutation(self, parent: tp.ArrayLike, max_ratio: float = 1.) -> tp.ArrayLike:
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
        max_mutations = max(2, int(max_ratio * dimension))
        p = 1. / np.arange(1, max_mutations)**1.5
        p /= np.sum(p)
        u = self.random_state.choice(np.arange(1, max_mutations), p=p)
        return self.portfolio_discrete_mutation(parent, u=u)

    def portfolio_discrete_mutation(self, parent: tp.ArrayLike, u: tp.Optional[int] = None) -> tp.ArrayLike:
        """Mutation discussed in
        https://arxiv.org/pdf/1606.05551v1.pdf
        We mutate a randomly drawn number of variables in average.
        """
        dimension = len(parent)
        if u is None:
            u = 1 if dimension == 1 else int(self.random_state.randint(1, dimension))
        if dimension == 1:  # corner case.
            return self.random_state.normal(0., 1., size=1)  # type: ignore
        boolean_vector = [True for _ in parent]
        while all(boolean_vector) and dimension != 1:
            boolean_vector = [self.random_state.rand() > (float(u) / dimension) for _ in parent]
        return [s if b else self.random_state.normal(0., 1.) for (b, s) in zip(boolean_vector, parent)]

    def discrete_mutation(self, parent: tp.ArrayLike) -> tp.ArrayLike:
        dimension = len(parent)
        boolean_vector = [True for _ in parent]
        while all(boolean_vector):
            boolean_vector = [self.random_state.rand() > (1. / dimension) for _ in parent]
        return [s if b else self.random_state.normal(0., 1.) for (b, s) in zip(boolean_vector, parent)]

    def crossover(self, parent: tp.ArrayLike, donor: tp.ArrayLike) -> tp.ArrayLike:
        mix = [self.random_state.choice([d, p]) for (p, d) in zip(parent, donor)]
        return self.discrete_mutation(mix)

    def get_roulette(self, archive: utils.Archive[utils.MultiValue], num: tp.Optional[int] = None) -> tp.Any:
        """Apply a roulette tournament selection.
        """
        if num is None:
            num = int(.999 + np.sqrt(len(archive)))
        # the following sort makes the line deterministic, and function seedable, at the cost of complexity!
        my_keys = sorted(archive.bytesdict.keys())
        my_keys_indices = self.random_state.choice(len(my_keys), size=min(num, len(my_keys)), replace=False)
        my_keys = [my_keys[i] for i in my_keys_indices]
        # best pessimistic value in a random set of keys
        return np.frombuffer(min(my_keys, key=lambda x: archive.bytesdict[x].pessimistic_confidence_bound))
