# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import itertools
import os
import platform
import typing as tp
from pathlib import Path
from unittest import SkipTest
import pytest
import numpy as np
from nevergrad.optimization import registry as optregistry
from nevergrad.functions.base import UnsupportedExperiment
from nevergrad.functions.mlda import datasets
from nevergrad.functions import rl
from nevergrad.common import testing

# from nevergrad.common.tools import Selector
from nevergrad.common import tools
from .xpbase import Experiment
from .utils import Selector
from . import experiments
from . import optgroups


@testing.parametrized(**{name: (name, maker) for name, maker in experiments.registry.items()})
def test_experiments_registry(name: str, maker: tp.Callable[[], tp.Iterator[experiments.Experiment]]) -> None:
    # Our PGAN is not well accepted by circleci.
    if "_pgan" in name and os.environ.get("CIRCLECI", False):
        raise SkipTest("Too slow in CircleCI")

    # Our IQAs and our ScikitLearn are not well guaranteed on Windows.
    if all(x in name for x in ["image", "quality"]) and platform.system() == "Windows":
        raise SkipTest("Image quality not guaranteed on Windows.")

    # Basic test.
    with tools.set_env(NEVERGRAD_PYTEST=1):
        with datasets.mocked_data():  # mock mlda data that should be downloaded
            check_maker(maker)  # this is to extract the function for reuse if other external packages need it

    # Some tests are skipped on CircleCI (but they do work well locally, if memory enough).
    if os.environ.get("CIRCLECI", False):
        if any(name == x for x in ["images_using_gan", "mlda", "realworld"]):
            raise SkipTest("Too slow in CircleCI")

    check_experiment(
        maker,
        "mltuning" in name,
        skip_seed=(name in ["rocket", "images_using_gan"]) or any(x in name for x in ["tuning", "image_"]),
    )  # this is a basic test on first elements, do not fully rely on it


@pytest.fixture(scope="module")  # type: ignore
def recorder() -> tp.Generator[tp.Dict[str, tp.List[optgroups.Optim]], None, None]:
    record: tp.Dict[str, tp.List[optgroups.Optim]] = {}
    yield record
    groups = sorted(record.items())
    string = "\n\n".join(f"{x} = {repr(y)}" for x, y in groups)
    filepath = Path(__file__).with_name("optimizer_groups.txt")
    filepath.write_text(string)


# pylint: disable=redefined-outer-name
@pytest.mark.parametrize("name", optgroups.registry)  # type: ignore
def test_groups_registry(name: str, recorder: tp.Dict[str, tp.List[optgroups.Optim]]) -> None:
    maker = optgroups.registry[name]
    opts = list(maker())
    for opt in opts:
        if isinstance(opt, str):
            assert opt in optregistry, f"{opt} is not registered."
    recorder[name] = opts


def check_maker(maker: tp.Callable[[], tp.Iterator[experiments.Experiment]]) -> None:
    generators = [maker() for _ in range(2)]
    # check 1 sample
    try:
        sample = next(maker())
    except UnsupportedExperiment as e:
        raise SkipTest("Skipping because unsupported") from e
    assert isinstance(sample, experiments.Experiment)
    # check names, coherence and non-randomness
    for k, (elem1, elem2) in enumerate(itertools.zip_longest(*generators)):
        assert not elem1.is_incoherent, f"Incoherent settings should be filtered out from generator:\n{elem1}"
        try:
            assert elem1 == elem2  # much faster but lacks explicit message
        except AssertionError:
            testing.printed_assert_equal(
                elem1.get_description(),
                elem2.get_description(),
                err_msg=f"Two paths on the generator differed (see element #{k})\n"
                "Generators need to be deterministic in order to split the workload!",
            )


def check_experiment(maker: tp.Any, short: bool = False, skip_seed: bool = False) -> None:
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
    rl.agents.TorchAgentFunction._num_test_evaluations = 1  # patch for faster evaluation
    for seed in [random_seed] if skip_seed else [random_seed, random_seed, random_seed + 1]:
        print(f"\nStarting with {seed % 100}")  # useful debug info when this test fails
        xps = list(itertools.islice(maker(seed), 0, 1 if short else 2))
        simplified = [
            Experiment(
                xp.function, algo, budget=2, num_workers=min(2, xp.optimsettings.num_workers), seed=xp.seed
            )
            for xp in xps
        ]
        np.random.shuffle(simplified)  # compute in any order
        selector = Selector(data=[xp.run() for xp in simplified])
        results.append(Selector(selector.loc[:, ["loss", "seed", "error"]]))  # elapsed_time can vary...
        assert results[-1].unique("error") == {""}, f"An error was raised during optimization:\n{results[-1]}"
    if skip_seed:
        return
    results[0].assert_equivalent(results[1], f"Non identical outputs for seed={random_seed}")
    np.testing.assert_raises(
        AssertionError,
        results[1].assert_equivalent,
        results[2],
        f"Identical output with different seeds (seed={random_seed})",
    )
