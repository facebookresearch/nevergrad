# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional
from ..functions import ArtificialFunction
from .. import optimization
from .xpbase import registry
from .xpbase import create_seed_generator
from .xpbase import Experiment
# pylint: disable=stop-iteration-return, too-many-nested-blocks


# %% CEC 2019 ## ## ## ## ## ## ## ## ## ## ##

@registry.register
def deceptivecec(seed: Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["deceptivemultimodal", "deceptiveillcond", "deceptivepath"]
    optims = ["PSO", "MiniQrDE", "MiniLhsDE", "MiniDE", "CMA", "QrDE", "DE", "LhsDE"]
    optims.append("CustomOptimizer")
    functions = [ArtificialFunction(name, block_dimension=2, num_blocks=n_blocks, rotation=rotation,
                                    aggregator=aggregator)
                 for name in names for rotation in [False, True] for n_blocks in [1, 2, 8, 16] for
                 aggregator in ["sum", "max"]]
    for func in functions:
        for optim in optims:
            for budget in [25,37,50,75,87] + list(range(100, 3001, 100)):
                 yield Experiment(func.duplicate(), optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def parallelcec(seed: Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = ["ScrHammersleySearch", "CMA", "PSO", "NaiveTBPSA", "OnePlusOne", "DE", "TwoPointsDE"]
    optims.append("CustomOptimizer")
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [25] for uv_factor in [0, 5]]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func.duplicate(), optim, budget=budget, num_workers=int(budget/5), seed=next(seedg))


@registry.register
def oneshotcec(seed: Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = sorted(x for x, y in optimization.registry.items() if y.one_shot)
    optims.append("CustomOptimizer")
    functions = [ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor) for name in names for bd in [3, 25] for uv_factor in [0, 5]]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func.duplicate(), optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def illcondicec(seed: Optional[int] = None) -> Iterator[Experiment]:
    """All optimizers on ill cond problems
    """
    seedg = create_seed_generator(seed)
    optims = ["CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE", "OnePlusOne", "SQP", "Cobyla", "Powell", "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE"]
    optims.append("CustomOptimizer")
    functions = [ArtificialFunction(name, block_dimension=50,
                 rotation=rotation) for name in ["cigar", "ellipsoid"]
                 for rotation in [True, False]]
    for optim in optims:
        for function in functions:
            for budget in [400, 4000, 40000]:
                yield Experiment(function.duplicate(), optim,
                    budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def noisycec(seed: Optional[int] = None) -> Iterator[Experiment]:
    """All optimizers on ill cond problems
    """
    seedg = create_seed_generator(seed)
    optims = ["FastGAOptimisticNoisyDiscreteOnePlusOne", "TBPSA", "NoisyBandit",
              "DoubleFastGAOptimisticNoisyDiscreteOnePlusOne", "SPSA", "NoisyOnePlusOne",
              "RandomSearch", "PortfolioOptimisticNoisyDiscreteOnePlusOne",
              "NoisyDiscreteOnePlusOne", "RandomScaleRandomSearch", "PortfolioNoisyDiscreteOnePlusOne"]
    optims.append("CustomOptimizer")
    for budget in [50000]:
        for optim in optims:
          for d in [2, 20, 200]:
            for rotation in [True]:
                for name in ["sphere", "rosenbrock"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(name=name, rotation=rotation, block_dimension=d, noise_level=10, noise_dissymmetry=noise_dissymmetry, translation_factor=1.)
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


