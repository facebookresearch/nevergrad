# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import itertools
from unittest.mock import patch
from typing import Callable, Iterator, Any
import numpy as np
from ..functions.mlda import datasets
from ..common import testing
from ..common.tools import Selector
from .xpbase import Experiment
from . import experiments


@testing.parametrized(**{name: (name, maker,) for name, maker in experiments.registry.items()})
def test_experiments_registry(name: str, maker: Callable[[], Iterator[experiments.Experiment]]) -> None:
    with patch("shutil.which", return_value="here"):  # do not check for missing packages
        with datasets.mocked_data():  # mock mlda data that should be downloaded
            check_maker(maker)  # this is to extract the function for reuse if other external packages need it
        if "mlda" not in name:
            check_seedable(maker)  # this is a basic test on first elements, do not fully rely on it


def check_maker(maker: Callable[[], Iterator[experiments.Experiment]]) -> None:
    generators = [maker() for _ in range(2)]
    # check 1 sample
    sample = next(maker())
    assert isinstance(sample, experiments.Experiment)
    # check names, coherence and non-randomness
    for k, (elem1, elem2) in enumerate(itertools.zip_longest(*generators)):
        assert not elem1.is_incoherent, f'Incoherent settings should be filtered out from generator:\n{elem1}'
        try:
            assert elem1 == elem2  # much faster but lacks explicit message
        except AssertionError:
            testing.printed_assert_equal(
                elem1.get_description(), elem2.get_description(), err_msg=f"Two paths on the generator differed (see element #{k})\n"
                "Generators need to be deterministic in order to split the workload!")


def check_seedable(maker: Any) -> None:
    """Randomized check of seedability for 8 first elements
    This test does not prove the complete seedability of the generator!  (would be way too slow)
    """
    # we use "maker: Any" because signature for one or the other case (seedable or not) is way too complex, and won't help much here...
    random_seed = np.random.randint(1000)
    signature = inspect.signature(maker)
    if not signature.parameters:
        return  # not designed to be seedable
    # draw twice with same random seed_generator and once with a different one
    results = []
    algo = "OnePlusOne"  # for simplifying the test
    for seed in [random_seed, random_seed, random_seed + 1]:
        xps = list(itertools.islice(maker(seed), 0, 8))
        simplified = [Experiment(xp.function, algo, budget=2, num_workers=min(2, xp.optimsettings.num_workers), seed=xp.seed)
                      for xp in xps]
        np.random.shuffle(simplified)  # compute in any order
        selector = Selector(data=[xp.run() for xp in simplified])
        results.append(Selector(selector.loc[:, ["loss", "seed"]]))  # elapsed_time can vary...
    results[0].assert_equivalent(results[1], f"Non identical outputs for seed={random_seed}")
    np.testing.assert_raises(AssertionError, results[1].assert_equivalent, results[2],
                             f"Identical output with different seeds (seed={random_seed})")
