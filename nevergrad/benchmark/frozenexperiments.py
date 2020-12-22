# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad import optimizers
from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.optimization import experimentalvariants  # pylint: disable=unused-import
from nevergrad.functions import ArtificialFunction
from .xpbase import registry
from .xpbase import create_seed_generator
from .xpbase import Experiment

# pylint: disable=stop-iteration-return, too-many-nested-blocks


@registry.register
def basic(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Test settings"""
    seedg = create_seed_generator(seed)
    function = ArtificialFunction(name="sphere", block_dimension=2, noise_level=1)
    np.random.seed(seed)  # seed before initializing the function!
    # initialization uses randomness
    function.transform_var._initialize()
    return iter([Experiment(function, optimizer="OnePlusOne", num_workers=2, budget=4, seed=next(seedg))])


@registry.register
def repeated_basic(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Test settings"""
    seedg = create_seed_generator(seed)
    function = ArtificialFunction(name="sphere", block_dimension=2, noise_level=1)
    optims: tp.List[tp.Union[str, ConfiguredOptimizer]] = ["OnePlusOne", optimizers.DifferentialEvolution()]
    for _ in range(5):
        for optim in optims:
            yield Experiment(function, optimizer=optim, num_workers=2, budget=4, seed=next(seedg))


@registry.register
def illcond(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All optimizers on ill cond problems"""
    seedg = create_seed_generator(seed)
    for budget in [500, 1000, 2000, 4000]:
        for optim in ["SQP", "DE", "CMA", "PSO", "RotationInvariantDE", "NelderMead"]:
            for rotation in [True, False]:
                for name in ["ellipsoid", "cigar"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=100)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def compabasedillcond(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All optimizers on ill cond problems"""
    seedg = create_seed_generator(seed)
    for budget in [500, 1000, 2000, 4000, 8000]:
        for optim in [
            "DE",
            "CMA",
            "PSO",
            "BPRotationInvariantDE",
            "RotationInvariantDE",
            "AlmostRotationInvariantDE",
            "AlmostRotationInvariantDEAndBigPop",
        ]:
            for rotation in [True, False]:
                for name in ["ellipsoid", "cigar"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=30)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def noise(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All optimizers on ill cond problems"""
    seedg = create_seed_generator(seed)
    optims = sorted(
        x
        for x, y in optimizers.registry.items()
        if ("TBPSA" in x or "ois" in x or "CMA" in x or "epea" in x) and "iscr" not in x
    )
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "cigar", "sphere4"]:
                    function = ArtificialFunction(
                        name=name, rotation=rotation, block_dimension=20, noise_level=10
                    )
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def dim10_smallbudget(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["sphere"]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [
        ArtificialFunction(
            name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks
        )
        for name in names
        for bd in [10]
        for uv_factor in [0]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [4, 8, 16, 32]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
# 2 variables matter - Scrambled Hammersley rules.
def dim10_select_two_features(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["sphere"]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [
        ArtificialFunction(
            name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks
        )
        for name in names
        for bd in [2]
        for uv_factor in [5]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [4, 8, 16, 32]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def dim10_select_one_feature(
    seed: tp.Optional[int] = None,
) -> tp.Iterator[Experiment]:  # One and only one variable matters - LHS wins.
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["sphere"]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [
        ArtificialFunction(
            name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks
        )
        for name in names
        for bd in [1]
        for uv_factor in [10]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [8, 10, 12, 14, 16, 18, 20]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def doe_dim4(
    seed: tp.Optional[int] = None,
) -> tp.Iterator[Experiment]:  # Here, QR performs best, then Random, then LHS, then Cauchy.
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["sphere"]  # n for n in ArtificialFunction.list_sorted_function_names() if "sphere" in n]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [
        ArtificialFunction(
            name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks
        )
        for name in names
        for bd in [4]
        for uv_factor in [0]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def oneshot4(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # General experiment comparing one-shot optimizers, excluding those with "large" or "small"
    # in the name.
    seedg = create_seed_generator(seed)
    names = ["sphere", "cigar", "ellipsoid", "rosenbrock", "rastrigin"]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [
        ArtificialFunction(
            name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks
        )
        for name in names
        for bd in [1, 4, 20]
        for uv_factor in [0, 10]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def oneshot3(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # General experiment comparing one-shot optimizers, excluding those with "large" or "small"
    # in the name.
    seedg = create_seed_generator(seed)
    names = ["sphere", "altcigar", "cigar", "ellipsoid", "rosenbrock", "rastrigin", "altellipsoid"]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [ArtificialFunction(name, block_dimension=bd) for name in names for bd in [4, 20]]
    for func in functions:
        for optim in optims:
            for budget in [30, 60, 100]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def oneshot2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # Experiment comparing one-shot optimizers in the context of useless vars vs critical vars.
    seedg = create_seed_generator(seed)
    names = ["sphere", "altcigar", "cigar", "ellipsoid", "rosenbrock", "rastrigin", "altellipsoid"]
    optims = sorted(
        x for x, y in optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x
    )
    functions = [
        ArtificialFunction(name, block_dimension=2, num_blocks=1, useless_variables=20) for name in names
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 60, 100]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def oneshot1(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Comparing one-shot optimizers as initializers for Bayesian Optimization."""
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:  # , 4000, 8000, 16000, 32000]:
        for optim in sorted(x for x, y in optimizers.registry.items() if "BO" in x):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:  # , "hm"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def metanoise(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    optims = ["NoisyBandit", "TBPSA", "NaiveTBPSA"]
    for budget in [15, 31, 62, 125, 250, 500, 1000, 2000, 4000, 8000]:
        for optim in optims:
            for noise_dissymmetry in [False, True]:
                function = ArtificialFunction(
                    name="sphere",
                    rotation=True,
                    block_dimension=1,
                    noise_level=10,
                    noise_dissymmetry=noise_dissymmetry,
                    translation_factor=10.0,
                )
                yield Experiment(function, optim, budget=budget, seed=next(seedg))
