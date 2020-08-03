# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from ..common import testing
from . import sequences
from .sequences import samplers


def test_get_first_primes() -> None:
    # check first 10
    first_10 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for k in range(11):
        np.testing.assert_array_equal(sequences._get_first_primes(k), first_10[:k])
    # generate a big number of them
    num = np.random.randint(1000, 1000000)
    output = sequences._get_first_primes(num)
    # verify the last value and one random value
    for value in [np.random.choice(output), output[-1]]:
        for k in range(3, 1 + int(np.sqrt(value)), 2):
            assert value % k, f"Value {value} obtained with num={num} can be divided by {k}"


@testing.parametrized(**{name: (name, sampler,) for name, sampler in samplers.items()})
def test_samplers(name: str, sampler_cls: tp.Type[sequences.Sampler]) -> None:
    sampler = sampler_cls(144, 4)
    np.testing.assert_equal(sampler.index, 0)
    output = sampler()
    np.testing.assert_equal(sampler.index, 1)
    np.testing.assert_equal(len(output), 144)
    assert min(output) > 0
    assert max(output) < 1


def test_sampler_draw() -> None:
    sampler = sequences.RandomSampler(5, 4)
    sampler.draw()


@testing.parametrized(
    # lhs=("LHSSampler", [0.069, 0.106, 0.384], [0.282, 0.857, 0.688]),  # previous: for the record
    lhs=("LHSSampler", [0.931, 0.422, 0.391], [0.428, 0.625, 0.797]),
    halton=("HaltonSampler", [0.5, 0.333, 0.2], [0.25, 0.667, 0.4]),
)
def test_sampler_values(name: str, seq1: tp.List[float], seq2: tp.List[float]) -> None:
    budget = 4
    np.random.seed(12)
    sampler = samplers[name](3, budget)
    samples = list(sampler)
    for k, expected in enumerate([seq1, seq2]):
        np.testing.assert_almost_equal(samples[k], expected, decimal=3)
    np.testing.assert_raises(AssertionError, sampler)  # budget is over
    sampler.reinitialize()
    samples2 = list(sampler)
    testing.printed_assert_equal(samples2, samples)


def test_permutation_generator() -> None:
    np.random.seed(12)
    gen = sequences.HaltonPermutationGenerator(5, scrambling=True)
    value1 = list(gen.get_permutations_generator())
    value2 = list(gen.get_permutations_generator())
    gen = sequences.HaltonPermutationGenerator(5, scrambling=True)
    value3 = list(gen.get_permutations_generator())
    testing.printed_assert_equal(value1, value2)
    # testing.printed_assert_equal(value1[:3], [[0, 1], [0, 2, 1], [0, 4, 3, 2, 1]])  # previous: for the record
    testing.printed_assert_equal(value1[:3], [[0, 1], [0, 1, 2], [0, 2, 3, 1, 4]])
    np.testing.assert_raises(AssertionError, testing.printed_assert_equal, value3, value2)
    #
    gen = sequences.HaltonPermutationGenerator(5, scrambling=False)
    value = list(gen.get_permutations_generator())
    testing.printed_assert_equal(value, [np.arange(p) for p in [2, 3, 5, 7, 11]])


def test_rescaler_on_hammersley() -> None:
    np.random.seed(12)
    sampler = sequences.HammersleySampler(dimension=3, budget=4, scrambling=True)
    samples = list(sampler)
    sampler.reinitialize()
    samples2 = list(sampler)
    sampler.reinitialize()
    np.testing.assert_array_equal(samples, samples2, "Not repeatable")  # test repeatability of hammersley first
    rescaler = sequences.Rescaler(sampler)
    sampler.reinitialize()
    rescaled_samples = [rescaler.apply(x) for x in sampler]
    expected = [[1e-15, 0.600, 0.667], [0.333, 0.200, 0.167], [0.667, 1.0, 1e-15], [1.0, 1e-15, 1.0]]
    np.testing.assert_almost_equal(rescaled_samples, expected, decimal=3)
