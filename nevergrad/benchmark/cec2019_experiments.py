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
def noisycec(seed: Optional[int] = None) -> Iterator[Experiment]:
    """All optimizers on ill cond problems
    """
    seedg = create_seed_generator(seed)
    optims = ["CustomOptimizer", "TBPSA", "NaiveTBPSA", "CMA", "NoisyBandit", "NoisyDE", "OptimisticNoisyOnePlusOne"]
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "cigar", "sphere4"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=20, noise_level=10)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def oneshotcec(seed: Optional[int] = None) -> Iterator[Experiment]:     # 2 variables matter - Scrambled Hammersley rules.
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optims = ["CustomOptimizer"] + sorted(x for x, y in optimization.registry.items() if y.one_shot and "arg" not in x and "mal" not in x)
    # functions are not initialized and duplicated at yield time, they will be initialized in the experiment (no need to seed here)
    for block_dimension in [1, 2, 4]:
        for useless_coeff in [0, 10]:
            for optim in optims:
                for budget in [4, 8, 16, 32, 128, 512, 2048, 8192]:
                    function = ArtificialFunction("sphere", block_dimension=block_dimension,
                                                  useless_variables=block_dimension * useless_coeff)
                    yield Experiment(function, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def parallelcec(seed: Optional[int] = None) -> Iterator[Experiment]:     # 2 variables matter - Scrambled Hammersley rules.
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optims = ["CustomOptimizer", "TwoPointsDE", "RandomSearch", "TBPSA", "CMA", "NaiveTBPSA", "PortfolioNoisyDiscreteOnePlusOne"]
    # functions are not initialized and duplicated at yield time, they will be initialized in the experiment (no need to seed here)
    for budget in [4, 8, 16, 32, 128, 512, 2048, 8192]:
        for block_dimension in [1, 2, 4]:
            for useless_coeff in [0, 10]:
                for optim in optims:
                    for name in ["sphere"]:
                        for num_workers in [4, 16, 64]:
                            function = ArtificialFunction(name, block_dimension=block_dimension,
                                                          useless_variables=block_dimension * useless_coeff)
                            yield Experiment(function, optim, budget=budget, num_workers=num_workers, seed=next(seedg))
