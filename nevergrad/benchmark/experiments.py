# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator, Optional
from ..functions import ArtificialFunction
from .. import optimization
from .xpbase import Experiment
from .xpbase import create_seed_generator
from .xpbase import registry
# register all frozen experiments
from . import frozenexperiments  # pylint:disable=unused-import
# pylint: disable=stop-iteration-return


@registry.register
def discrete(seed: Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = [n for n in ArtificialFunction.list_sorted_function_names() if "one" in n or "jump" in n]
    optims = sorted(x for x, y in optimization.registry.items() if "iscrete" in x and "epea" not in x and "DE" not in x
                    and "SSNEA" not in x)
    functions = [ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
                 for name in names for bd in [30] for uv_factor in [0, 5, 10] for n_blocks in [1]]
    # functions are not initialized and duplicated at yield time, they will be initialized in the experiment (no need to seed here)
    for func in functions:
        for optim in optims:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
                           1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
                           2700, 2800, 2900, 3000]:  # , 10000]:
                yield Experiment(func.duplicate(), optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def minidoe(seed: Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    names = ["sphere"]
    optims = sorted(x for x, y in optimization.registry.items() if y.one_shot and "arg" not in x and "mal" not in x)
    functions = [ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
                 for name in names for bd in [1, 4, 20] for uv_factor in [0, 10, 100] for n_blocks in [1]]
    # functions are not initialized and duplicated at yield time, they will be initialized in the experiment
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                # duplicate -> each Experiment has different randomness
                yield Experiment(func.duplicate(), optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def doe_dim10(seed: Optional[int] = None) -> Iterator[Experiment]:  # LHS performs best, followed by QR and random
    # nearly equally (Hammersley better than random, Halton not clearly; scrambling improves results).
    # prepare list of parameters to sweep for independent variables
    names = ["sphere"]
    seedg = create_seed_generator(seed)
    optims = sorted(x for x, y in optimization.registry.items() if y.one_shot and "arg" not in x and "mal" not in x)
    functions = [ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
                 for name in names for bd in [10] for uv_factor in [0] for n_blocks in [1]]
    # functions are not initialized and duplicated at yield time, they will be initialized in the experiment (no need to seed here)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                # duplicate -> each Experiment has different randomness
                yield Experiment(func.duplicate(), optim, budget=budget, num_workers=1, seed=next(seedg))



@registry.register
def spsa_benchmark(seed: Optional[int] = None) -> Iterator[Experiment]:
    """All optimizers on ill cond problems. This benchmark is based on the noise benchmark.
    """
    seedg = create_seed_generator(seed)
    optims = sorted(x for x, y in optimization.registry.items()
                    if (any(e in x for e in "TBPSA SPSA".split())
                        and "iscr" not in x))
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=20, noise_level=10)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))
