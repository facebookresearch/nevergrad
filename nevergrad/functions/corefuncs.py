# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Dict, Any, Tuple, List, Callable
import numpy as np
from .utils import PostponedObject
from ..instrumentation import discretization
from ..common.decorators import Registry


registry = Registry[Callable[[np.ndarray], float]]()


def _onemax(x: List[int]) -> float:
    return len(x) - sum(1 if int(round(w)) == 1 else 0 for w in x)


def _leadingones(x: List[int]) -> float:
    for i, x_ in enumerate(list(x)):
        if int(round(x_)) != 1:
            return len(x) - i
    return 0


def _jump(x: List[int]) -> float:  # TODO: docstring?
    n = len(x)
    m = n // 4
    o = n - _onemax(x)
    if o == n or o <= n - m:
        return n - m - o
    return o  # Deceptive part.


def _styblinksitang(x: np.ndarray, noise: float) -> float:
    x = np.asarray(x)
    val = np.sum(np.power(x, 4) - 16 * np.power(x, 2) + 5 * x)
    # return a positive value for maximization
    return float(39.16599 * len(x) + 1 * 0.5 * val + noise * np.random.normal(size=val.shape))


@registry.register
def delayedsphere(x: np.ndarray) -> float:
    '''For asynchronous experiments, we induce delays.'''
    time.sleep(abs(1./x[0]) / 100000. if x[0] != 0. else 0.)
    return float(np.sum(x**2))


class DelayedSphere(PostponedObject):

    def __call__(self, x: np.ndarray) -> float:
        return float(np.sum(x**2))

    def get_postponing_delay(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], value: float) -> float:
        x = args[0]
        return float(abs(1./x[0]) / 1000.) if x[0] != 0. else 0.


registry.register(DelayedSphere())


@registry.register
def sphere(x: np.ndarray) -> float:
    return float(np.sum(x**2))


@registry.register
def sphere1(x: np.ndarray) -> float:
    return float(np.sum((x - 1.)**2))


@registry.register
def sphere2(x: np.ndarray) -> float:
    return float(np.sum((x - 2.)**2))


@registry.register
def sphere4(x: np.ndarray) -> float:
    return float(np.sum((x - 4.)**2))


@registry.register
def maxdeceptive(x: np.ndarray) -> float:
    dec = 3 * x**2 - (2 / (3**(x - 2)**2 + .1))
    return float(np.max(dec))


@registry.register
def sumdeceptive(x: np.ndarray) -> float:
    dec = 3 * x**2 - (2 / (3**(x - 2)**2 + .1))
    return float(np.sum(dec))


@registry.register
def altcigar(x: np.ndarray) -> float:
    return float(x[-1]**2 + 1000000. * np.sum(x[:-1]**2))


@registry.register
def cigar(x: np.ndarray) -> float:
    return float(x[0]**2 + 1000000. * np.sum(x[1:]**2))


@registry.register
def altellipsoid(y: np.ndarray) -> float:
    x = y[::-1]
    return sum((10**(6 * (i - 1) / float(len(x) - 1))) * (x[i]**2) for i in range(len(x)))


@registry.register
def ellipsoid(x: np.ndarray) -> float:
    return sum((10**(6 * (i - 1) / float(len(x) - 1))) * (x[i]**2) for i in range(len(x)))


@registry.register
def rastrigin(x: np.ndarray) -> float:
    cosi = float(np.sum(np.cos(2 * np.pi * x)))
    return float(10 * (len(x) - cosi) + sphere(x))


@registry.register
def hm(x: np.ndarray) -> float:
    return float(np.sum((x**2) * (1.1 + np.cos(1. / x))))


@registry.register
def rosenbrock(x: np.ndarray) -> float:
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


@registry.register
def griewank(x: np.ndarray) -> float:
    part1 = np.sum(x**2)
    part2 = np.prod(np.cos(x / np.sqrt(1 + np.arange(len(x)))))
    return 1 + (float(part1)/4000.0) - float(part2)


@registry.register
def deceptiveillcond(x: np.ndarray) -> float:
    assert len(x) >= 2
    return float(max(np.abs(np.arctan(x[1]/x[0])),
                     np.sqrt(x[0]**2. + x[1]**2.),
                     1. if x[0] > 0 else 0.) if x[0] != 0. else float("inf"))


@registry.register
def deceptivepath(x: np.ndarray) -> float:
    assert len(x) >= 2
    distance = np.sqrt(x[0]**2 + x[1]**2)
    if distance == 0.:
        return 0.
    angle = np.arctan(x[0] / x[1]) if x[1] != 0. else np.pi / 2.
    invdistance = (1. / distance) if distance > 0. else 0.
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return 1.
    return float(distance)


@registry.register
def deceptivemultimodal(x: np.ndarray) -> float:
    assert len(x) >= 2
    distance = np.sqrt(x[0]**2 + x[1]**2)
    if distance == 0.:
        return 0.
    angle = np.arctan(x[0] / x[1]) if x[1] != 0. else np.pi / 2.
    invdistance = int(1. / distance) if distance > 0. else 0.
    if np.abs(np.cos(invdistance) - angle) > 0.1:
        return 1.
    return float(distance)


@registry.register
def lunacek(x: np.ndarray) -> float:
    """ Based on https://www.cs.unm.edu/~neal.holts/dga/benchmarkFunction/lunacek.html."""
    problemDimensions = len(x)
    s = 1.0 - (1.0 / (2.0 * np.sqrt(problemDimensions + 20.0) - 8.2))
    mu1 = 2.5
    mu2 = - np.sqrt(abs((mu1**2 - 1.0) / s))
    firstSum = 0.0
    secondSum = 0.0
    thirdSum = 0.0
    for i in range(problemDimensions):
        firstSum += (x[i]-mu1)**2
        secondSum += (x[i]-mu2)**2
        thirdSum += 1.0 - np.cos(2*np.pi*(x[i]-mu1))
    return min(firstSum, 1.0*problemDimensions + secondSum)+10*thirdSum


# following functions using discretization should not be used with translation/rotation


@registry.register_with_info(no_transfrom=True)
def hardonemax(y: np.ndarray) -> float:
    return _onemax(discretization.threshold_discretization(y))


@registry.register_with_info(no_transfrom=True)
def hardjump(y: np.ndarray) -> float:
    return _jump(discretization.threshold_discretization(y))


@registry.register_with_info(no_transfrom=True)
def hardleadingones(y: np.ndarray) -> float:
    return _leadingones(discretization.threshold_discretization(y))


@registry.register_with_info(no_transfrom=True)
def hardonemax5(y: np.ndarray) -> float:
    return _onemax(discretization.threshold_discretization(y, 5))


@registry.register_with_info(no_transfrom=True)
def hardjump5(y: np.ndarray) -> float:
    return _jump(discretization.threshold_discretization(y, 5))


@registry.register_with_info(no_transfrom=True)
def hardleadingones5(y: np.ndarray) -> float:
    return _leadingones(discretization.threshold_discretization(y, 5))


@registry.register_with_info(no_transfrom=True)
def onemax(y: np.ndarray) -> float:
    return _onemax(discretization.softmax_discretization(y))


@registry.register_with_info(no_transfrom=True)
def jump(y: np.ndarray) -> float:
    return _jump(discretization.softmax_discretization(y))


@registry.register_with_info(no_transfrom=True)
def leadingones(y: np.ndarray) -> float:
    return _leadingones(discretization.softmax_discretization(y))


@registry.register_with_info(no_transfrom=True)
def onemax5(y: np.ndarray) -> float:
    return _onemax(discretization.softmax_discretization(y, 5))


@registry.register_with_info(no_transfrom=True)
def jump5(y: np.ndarray) -> float:
    return _jump(discretization.softmax_discretization(y, 5))


@registry.register_with_info(no_transfrom=True)
def leadingones5(y: np.ndarray) -> float:
    return _leadingones(discretization.softmax_discretization(y, 5))


@registry.register_with_info(no_transfrom=True)
def genzcornerpeak(y: np.ndarray) -> float:
    value = float(1 + np.mean(np.tanh(y)))
    if value == 0:
        return float("inf")
    return value**(-len(y) - 1)


@registry.register_with_info(no_transfrom=True)
def minusgenzcornerpeak(y: np.ndarray) -> float:
    return -float(genzcornerpeak(y))


@registry.register
def genzgaussianpeakintegral(x: np.ndarray) -> float:
    return float(np.exp(-np.sum(x**2 / 4.)))


@registry.register
def minusgenzgaussianpeakintegral(x: np.ndarray) -> float:
    return -float(np.exp(-sum(x**2 / 4.)))


@registry.register
def slope(x: np.ndarray) -> float:
    return sum(x)


@registry.register
def linear(x: np.ndarray) -> float:
    return float(np.tanh(x[0]))


@registry.register
def st0(x: np.ndarray) -> float:
    return _styblinksitang(x, 0)


@registry.register
def st1(x: np.ndarray) -> float:
    return _styblinksitang(x, 1)


@registry.register
def st10(x: np.ndarray) -> float:
    return _styblinksitang(x, 10)


@registry.register
def st100(x: np.ndarray) -> float:
    return _styblinksitang(x, 100)
