# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Samplers in [0,1]^d.
"""

import numpy as np
from numpy.random import RandomState
import nevergrad.common.typing as tp
from nevergrad.common.decorators import Registry


samplers: Registry[tp.Type['Sampler']] = Registry()


def _get_first_primes(num: int) -> np.ndarray:
    """Computes the first num primes
    """
    # upper bound for the value of the n_th prime_number (n >= 6)
    # https://en.wikipedia.org/wiki/Prime_number_theorem
    # Possible optimization, only odd numbers in is_prime array?
    if num < 6:
        return np.array([2, 3, 5, 7, 11][: num], dtype=int)
    is_prime = np.ones(int(1 + num * (np.log(np.log(num)) + np.log(num))), dtype=bool)
    is_prime[[0, 1]] = 0
    for index in range(1 + int(np.sqrt(len(is_prime)))):
        if is_prime[index]:
            is_prime[index + index::index] = False
    primes = np.where(is_prime)[0]
    if len(primes) < num:
        raise RuntimeError(f"There is an error on the upper bound of the primes for num={num}")
    return primes[:num]  # type: ignore


class Sampler:

    def __init__(self, dimension: int, budget: tp.Optional[int] = None, random_state: tp.Optional[RandomState] = None) -> None:
        if random_state is None:
            random_state = np.random.RandomState(np.random.randint(2**32, dtype=np.uint32))
        self.random_state = random_state
        self.dimension = dimension
        self.budget = budget
        self.index = 0

    def _internal_sampler(self) -> tp.ArrayLike:
        raise NotImplementedError("Missing sampling function! which is quite necessary for a sampler.")

    def __call__(self) -> tp.ArrayLike:
        # TODO deprecate this method
        assert self.budget is None or self.index < self.budget, "Over the budget (reinitialize if you want to start over)"
        sample = self._internal_sampler()
        self.index += 1
        return sample

    def __iter__(self) -> tp.Iterator[tp.ArrayLike]:  # unused, but could be useful
        assert self.index == 0, "Reinitialize before iterating again"  # backward compatibility
        assert self.budget is not None, "Iterable does not work if budget is not specified"  # TODO make it work
        return (self() for _ in range(self.budget))

    def reinitialize(self) -> None:
        self.index = 0

    def draw(self) -> None:
        """Simple ASCII drawing of the sampling pattern (for testing/visualization purpose only)
        """
        sampler = self.__class__(self.dimension, budget=self.budget)  # create a clone instance to avoid border effects
        assert sampler.budget is not None
        sample = [sampler() for _ in range(sampler.budget)]
        for i in range(sampler.dimension):
            for j in range(i + 1, sampler.dimension):
                print("plotting coordinates " + str(i) + "," + str(j))
                tab = [["." for _ in range(80)] for _ in range(20)]
                for s in sample:
                    x = int(s[i] * 20)
                    y = int(s[j] * 80)
                    tab[x][y] = "*"
                for t in tab:
                    print("".join(t))


@samplers.register
class LHSSampler(Sampler):

    def __init__(self, dimension: int, budget: int, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None:
        if scrambling:
            raise ValueError("LHSSampler does not support scrambling")
        super().__init__(dimension, budget, random_state=random_state)
        self.permutations = np.zeros((dimension, budget), dtype=int)
        for k in range(dimension):
            self.permutations[k] = self.random_state.permutation(budget)
        self.seed = self.random_state.randint(2**32, dtype=np.uint32)
        self.randg = np.random.RandomState(self.seed)

    def reinitialize(self) -> None:
        super().reinitialize()
        self.randg = np.random.RandomState(self.seed)

    def _internal_sampler(self) -> tp.ArrayLike:
        x = self.permutations[:, self.index].tolist()
        assert self.budget is not None
        return (x + self.randg.uniform(size=self.dimension)) / float(self.budget)  # type: ignore


@samplers.register
class RandomSampler(Sampler):

    def __init__(self, dimension: int, budget: int, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None:
        if scrambling:
            raise ValueError("RandomSampler does not support scrambling")
        super().__init__(dimension, budget, random_state=random_state)

    def _internal_sampler(self) -> tp.ArrayLike:
        return self.random_state.uniform(0, 1, self.dimension)  # type: ignore


class HaltonPermutationGenerator:
    """Provides a light-memory access to a possibly huge list of permutations
    (at the cost of being slightly slower)
    """  # the bottleneck here is vdc anyway

    def __init__(self, dimension: int, scrambling: bool = False, random_state: tp.Optional[RandomState] = None) -> None:
        if random_state is None:
            random_state = np.random.RandomState(np.random.randint(2**32, dtype=np.uint32))
        self.dimension = dimension
        self.scrambling = scrambling
        self.primes = _get_first_primes(dimension).tolist()
        self.seed = random_state.randint(2**32, dtype=np.uint32)
        self.fulllist = np.arange(self.primes[-1]) if self.primes else []

    def get_permutations_generator(self) -> tp.Iterator[tp.ArrayLike]:
        if self.scrambling:
            randgen = np.random.RandomState(seed=self.seed)
            return (np.concatenate(([0], randgen.choice(self.fulllist[1: p], p - 1, replace=False)), axis=0) for p in self.primes)
        else:
            return (self.fulllist[:p] for p in self.primes)


@samplers.register
class HaltonSampler(Sampler):

    def __init__(self, dimension: int, budget: tp.Optional[int] = None, scrambling: bool = False,
                 random_state: tp.Optional[RandomState] = None) -> None:
        super().__init__(dimension, budget, random_state=random_state)
        self.permgen = HaltonPermutationGenerator(dimension, scrambling, random_state=random_state)

    def vdc(self, n: int, permut: tp.List[int]) -> float:  # TODO speed up?
        base = len(permut)  # should be a prime number
        vdc, denom = 0., 1
        n += 1
        while n:
            denom *= base
            n, remainder = divmod(n, base)
            remainder = permut[remainder]
            vdc += float(remainder) / float(denom)
        return vdc

    def _internal_sampler(self) -> tp.ArrayLike:
        # len(sigma) describes all n=dimension first prime numbers
        sample = [self.vdc(self.index, sigma) for sigma in self.permgen.get_permutations_generator()]  # type: ignore
        return sample


@samplers.register
class HammersleySampler(HaltonSampler):

    def __init__(self, dimension: int, budget: tp.Optional[int] = None, scrambling: bool = False,
                 random_state: tp.Optional[RandomState] = None) -> None:
        assert budget is not None
        super().__init__(dimension - 1, budget, scrambling, random_state=random_state)

    def _internal_sampler(self) -> tp.ArrayLike:
        assert self.budget is not None
        return np.concatenate(([(self.index + .5) / float(self.budget)], super()._internal_sampler()))  # type: ignore


class Rescaler:

    def __init__(self, points: tp.Iterable[tp.ArrayLike]) -> None:
        iterp = iter(points)
        self.sample_mins = np.array(next(iterp), copy=False)
        self.sample_maxs = self.sample_mins
        for point in iterp:
            self.sample_mins = np.minimum(self.sample_mins, point)
            self.sample_maxs = np.maximum(self.sample_maxs, point)
        self.epsilon = min([x for x in self.sample_mins] + [1 - s for s in self.sample_maxs] + [1e-15])
        assert self.epsilon > 0., f'Non-positive epsilon={self.epsilon} from mins {self.sample_mins} and maxs {self.sample_maxs}'

    def apply(self, point: tp.ArrayLike) -> np.ndarray:
        point = np.array(point, copy=False)
        factor = (1 - 2 * self.epsilon) / (self.sample_maxs - self.sample_mins)
        return self.epsilon + factor * (point - self.sample_mins)  # type: ignore
