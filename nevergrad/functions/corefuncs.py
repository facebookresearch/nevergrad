# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import exp, sqrt, tanh
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common.decorators import Registry


registry: Registry[tp.Callable[[np.ndarray], float]] = Registry()


class DiscreteFunction:
    def __init__(self, name: str, arity: int = 2) -> None:
        """Returns a classical discrete function for test, in the domain {0,1,...,arity-1}^d.
        The name can be onemax, leadingones, or jump.

        onemax(x) is the most classical case of discrete functions, adapted to minimization.
        It is originally designed for lists of bits. It just counts the number of 1,
        and returns len(x) - number of ones. However, the present function perturbates the location of the
        optimum, so that tests can not be easily biased by a wrong initialization. So the optimum,
        instead of being located at (1,1,...,1), is located at (0,1,2,...,arity-1,0,1,2,...).

        leadingones is the second most classical discrete function, adapted for minimization.
        Before perturbation of the location of the optimum as above,
        it returns len(x) - number of initial 1. I.e.
        leadingones([0 1 1 1]) = 4,
        leadingones([1 1 1 1]) = 0,
        leadingones([1 0 0 0]) = 3.
        The present Leadingones function uses a perturbation as documented above for OneMax: we count the number
        of initial correct values, a correct values being 0 for variable 1, 1 for variable 2, 2 for variable 3, and
        so on.

        There exists variants of jump functions: the principle of a jump function is that local descent does not succeed.
        Jumps are necessary. We are here in minimization, hence a formulation slightly different from most discrete optimization
        papers, which usually assume maximization. We use the same perturbation as detailed above for leadingones and onemax,
        i.e. the optimum is located at (0,1,2,...,arity-1,0,1,2,...).
        """
        self._arity = arity
        self._func = dict(onemax=self.onemax, leadingones=self.leadingones, jump=self.jump)[name]

    def __call__(self, x: tp.ArrayLike) -> float:
        return self._func(x)

    def onemax(self, x: tp.ArrayLike) -> float:
        diff = np.round(x) - (np.arange(len(x)) % self._arity)
        return float(np.sum(diff != 0))

    def leadingones(self, x: tp.ArrayLike) -> float:
        diff = np.round(x) - (np.arange(len(x)) % self._arity)
        nonzeros = np.nonzero(diff)[0]
        return float(len(x) - nonzeros[0] if nonzeros.size else 0)

    def jump(self, x: tp.ArrayLike) -> float:
        n = len(x)
        m = n // 4
        o = n - self.onemax(x)
        if o == n or o <= n - m:
            return n - m - o
        return o  # Deceptive part.


def _styblinksitang(x: np.ndarray, noise: float) -> float:
    """Classical function for testing noisy optimization."""
    x2 = x ** 2
    val = x2.dot(x2) + np.sum(5 * x - 16 * x2)
    # return a positive value for maximization
    return float(39.16599 * len(x) + 0.5 * val + noise * np.random.normal(size=val.shape))


class DelayedSphere:
    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    def compute_pseudotime(  # pylint: disable=unused-argument
        self, input_parameter: tp.Any, value: float
    ) -> float:
        x = input_parameter[0][0]
        return float(abs(1.0 / x[0]) / 1000.0) if x[0] != 0.0 else 0.0


registry.register(DelayedSphere())


@registry.register
def sphere(x: np.ndarray) -> float:
    """The most classical continuous optimization testbed.

    If you do not solve that one then you have a bug."""
    assert x.ndim == 1
    return float(x.dot(x))


@registry.register
def sphere1(x: np.ndarray) -> float:
    """Translated sphere function."""
    return sphere(x - 1.0)


@registry.register
def sphere2(x: np.ndarray) -> float:
    """A bit more translated sphere function."""
    return sphere(x - 2.0)


@registry.register
def sphere4(x: np.ndarray) -> float:
    """Even more translated sphere function."""
    return sphere(x - 4.0)


@registry.register
def maxdeceptive(x: np.ndarray) -> float:
    dec = 3 * x ** 2 - (2 / (3 ** (x - 2) ** 2 + 0.1))
    return float(np.max(dec))


@registry.register
def sumdeceptive(x: np.ndarray) -> float:
    dec = 3 * x ** 2 - (2 / (3 ** (x - 2) ** 2 + 0.1))
    return float(np.sum(dec))


@registry.register
def altcigar(x: np.ndarray) -> float:
    """Similar to cigar, but variables in inverse order.

    E.g. for pointing out algorithms not invariant to the order of variables."""
    return float(x[-1]) ** 2 + 1000000.0 * sphere(x[:-1])


@registry.register
def discus(x: np.ndarray) -> float:
    """Only one variable is very penalized."""
    return sphere(x[1:]) + 1000000.0 * float(x[0]) ** 2


@registry.register
def cigar(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.

    The other classical example is ellipsoid.
    """
    return float(x[0]) ** 2 + 1000000.0 * sphere(x[1:])


@registry.register
def bentcigar(x: np.ndarray) -> float:
    """Classical example of ill conditioned function, but bent."""
    y = np.asarray(
        [
            x[i] ** (1 + 0.5 * np.sqrt(x[i]) * (i - 1) / (len(x) - 1)) if x[i] > 0.0 else x[i]
            for i in range(len(x))
        ]
    )
    return float(y[0]) ** 2 + 1000000.0 * sphere(y[1:])


@registry.register
def multipeak(x: np.ndarray) -> float:
    """Inspired by M. Gallagher's Gaussian peaks function."""
    v = 10000.0
    for a in range(101):
        x_ = np.asarray([np.cos(a + np.sqrt(i)) for i in range(len(x))])
        v = min(v, a / 101.0 + np.exp(sphere(x - x_)))
    return v


@registry.register
def altellipsoid(y: np.ndarray) -> float:
    """Similar to Ellipsoid, but variables in inverse order.

    E.g. for pointing out algorithms not invariant to the order of variables."""
    return ellipsoid(y[::-1])


def step(s: float) -> float:
    return float(np.exp(int(np.log(s))))


@registry.register
def stepellipsoid(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.

    But we add a 'step', i.e. we set the gradient to zero everywhere.
    Compared to some existing testbeds, we decided to have infinitely many steps.
    """
    dim = x.size
    weights = 10 ** np.linspace(0, 6, dim)
    return float(step(weights.dot(x ** 2)))


@registry.register
def ellipsoid(x: np.ndarray) -> float:
    """Classical example of ill conditioned function.

    The other classical example is cigar.
    """
    dim = x.size
    weights = 10 ** np.linspace(0, 6, dim)
    return float(weights.dot(x ** 2))


@registry.register
def rastrigin(x: np.ndarray) -> float:
    """Classical multimodal function."""
    cosi = float(np.sum(np.cos(2 * np.pi * x)))
    return float(10 * (len(x) - cosi) + sphere(x))


@registry.register
def bucherastrigin(x: np.ndarray) -> float:
    """Classical multimodal function. No box-constraint penalization here."""
    s = np.asarray(
        [
            x[i] * (10 if x[i] > 0.0 and i % 2 else 1) * (10 ** ((i - 1) / (2 * (len(x) - 1))))
            for i in range(len(x))
        ]
    )
    cosi = float(np.sum(np.cos(2 * np.pi * s)))
    return float(10 * (len(x) - cosi) + sphere(s))


@registry.register
def doublelinearslope(x: np.ndarray) -> float:
    """We decided to use two linear slopes rather than having a constraint artificially added for
    not having the optimum at infinity."""
    return float(np.abs(np.sum(x)))


@registry.register
def stepdoublelinearslope(x: np.ndarray) -> float:
    return step(np.abs(np.sum(x)))


@registry.register
def hm(x: np.ndarray) -> float:
    """New multimodal function (proposed for Nevergrad)."""
    return float((x ** 2).dot(1.1 + np.cos(1.0 / x)))


@registry.register
def rosenbrock(x: np.ndarray) -> float:
    x_m_1 = x[:-1] - 1
    x_diff = x[:-1] ** 2 - x[1:]
    return float(100 * x_diff.dot(x_diff) + x_m_1.dot(x_m_1))


@registry.register
def ackley(x: np.ndarray) -> float:
    dim = x.size
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20.0 * exp(-0.2 * sqrt(sphere(x) / dim)) - exp(sum_cos / dim) + 20 + exp(1)


@registry.register
def schwefel_1_2(x: np.ndarray) -> float:
    cx = np.cumsum(x)
    return sphere(cx)


@registry.register
def griewank(x: np.ndarray) -> float:
    """Multimodal function, often used in Bayesian optimization."""
    part1 = sphere(x)
    part2 = np.prod(np.cos(x / np.sqrt(1 + np.arange(len(x)))))
    return 1 + (float(part1) / 4000.0) - float(part2)


@registry.register
def deceptiveillcond(x: np.ndarray) -> float:
    """An extreme ill conditioned functions. Most algorithms fail on this.

    The condition number increases to infinity as we get closer to the optimum."""
    assert len(x) >= 2
    return float(
        max(np.abs(np.arctan(x[1] / x[0])), np.sqrt(x[0] ** 2.0 + x[1] ** 2.0), 1.0 if x[0] > 0 else 0.0)
        if x[0] != 0.0
        else float("inf")
    )


@registry.register
def deceptivepath(x: np.ndarray) -> float:
    """A function which needs following a long path. Most algorithms fail on this.

    The path becomes thiner as we get closer to the optimum."""
    assert len(x) >= 2
    distance = np.sqrt(x[0] ** 2 + x[1] ** 2)
    if distance == 0.0:
        return 0.0
    angle = np.arctan(x[0] / x[1]) if x[1] != 0.0 else np.pi / 2.0
    invdistance = (1.0 / distance) if distance > 0.0 else 0.0
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return 1.0
    return float(distance)


@registry.register
def deceptivemultimodal(x: np.ndarray) -> float:
    """Infinitely many local optima, as we get closer to the optimum."""
    assert len(x) >= 2
    distance = np.sqrt(x[0] ** 2 + x[1] ** 2)
    if distance == 0.0:
        return 0.0
    angle = np.arctan(x[0] / x[1]) if x[1] != 0.0 else np.pi / 2.0
    invdistance = int(1.0 / distance) if distance > 0.0 else 0.0
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return 1.0
    return float(distance)


@registry.register
def lunacek(x: np.ndarray) -> float:
    """Multimodal function.

    Based on https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/lunacek.html."""
    problemDimensions = len(x)
    s = 1.0 - (1.0 / (2.0 * np.sqrt(problemDimensions + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = -np.sqrt(abs((mu1 ** 2 - 1.0) / s))
    firstSum = 0.0
    secondSum = 0.0
    thirdSum = 0.0
    for i in range(problemDimensions):
        firstSum += (x[i] - mu1) ** 2
        secondSum += (x[i] - mu2) ** 2
        thirdSum += 1.0 - np.cos(2 * np.pi * (x[i] - mu1))
    return min(firstSum, 1.0 * problemDimensions + secondSum) + 10 * thirdSum


@registry.register_with_info(no_transform=True)
def genzcornerpeak(y: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
    value = float(1 + np.mean(np.tanh(y)))
    if value == 0:
        return float("inf")
    return value ** (-len(y) - 1)


@registry.register_with_info(no_transform=True)
def minusgenzcornerpeak(y: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
    return -genzcornerpeak(y)


@registry.register
def genzgaussianpeakintegral(x: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
    return exp(-sphere(x) / 4.0)


@registry.register
def minusgenzgaussianpeakintegral(x: np.ndarray) -> float:
    """One of the Genz functions, originally used in integration,

    tested in optim because why not."""
    return -genzgaussianpeakintegral(x)


@registry.register
def slope(x: np.ndarray) -> float:
    return sum(x)


@registry.register
def linear(x: np.ndarray) -> float:
    return tanh(x[0])


@registry.register
def st0(x: np.ndarray) -> float:
    """Styblinksitang function with 0 noise."""
    return _styblinksitang(x, 0)


@registry.register
def st1(x: np.ndarray) -> float:
    """Styblinksitang function with noise 1."""
    return _styblinksitang(x, 1)


@registry.register
def st10(x: np.ndarray) -> float:
    """Styblinksitang function with noise 10."""
    return _styblinksitang(x, 10)


@registry.register
def st100(x: np.ndarray) -> float:
    """Styblinksitang function with noise 100."""
    return _styblinksitang(x, 100)
