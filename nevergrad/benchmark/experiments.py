# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from typing import Iterator, Optional, List, Union, Any
import numpy as np
import nevergrad as ng
from nevergrad.functions import ExperimentFunction
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import FarOptimumFunction
from nevergrad.functions import MultiobjectiveFunction
from nevergrad.functions import mlda as _mlda
from nevergrad.functions.photonics import Photonics
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions.powersystems import PowerSystem
from nevergrad.functions.stsp import STSP
from nevergrad.functions import rl
from nevergrad.functions.games import game
from .xpbase import Experiment as Experiment
from .xpbase import create_seed_generator
from .xpbase import registry as registry  # noqa

# register all frozen experiments
from . import frozenexperiments  # noqa # pylint: disable=unused-import

# pylint: disable=stop-iteration-return, too-many-nested-blocks, too-many-locals, line-too-long
# for black (since lists are way too long...):
# fmt: off


# Discrete functions on {0,1}^d.
@registry.register
def discrete2(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Discrete test bed, including useless variables, binary only.
    Poorly designed, should be reimplemented from scratch using a decent instrumentation."""
    seedg = create_seed_generator(seed)
    names = [n for n in ArtificialFunction.list_sorted_function_names()
             if ("one" in n or "jump" in n) and ("5" not in n) and ("hard" in n)]
    optims = ["NGO", "Shiva", "DiagonalCMA", "CMA"] + sorted(
        x for x, y in ng.optimizers.registry.items() if "andomSearch" in x or "PBIL" in x or "cGA" in x or
        ("iscrete" in x and "epea" not in x and "DE" not in x and "SSNEA" not in x)
    )
    functions = [
        ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
        for name in names
        for bd in [5, 30, 180]
        for uv_factor in [0, 5, 10]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for nw in [1, 10]:
                for budget in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                               1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]:  # , 10000]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def discrete(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Discrete test bed, including useless variables, 5 values or 2 values per character.
    Poorly designed, should be reimplemented from scratch using a decent instrumentation."""
    seedg = create_seed_generator(seed)
    names = [n for n in ArtificialFunction.list_sorted_function_names() if "one" in n or "jump" in n]
    optims = ["NGO", "Shiva", "DiagonalCMA", "CMA"] + sorted(
        x for x, y in ng.optimizers.registry.items()
        if "andomSearch" in x or ("iscrete" in x and "epea" not in x and "DE" not in x and "SSNEA" not in x)
    )
    # Block dimension = dimension of a block on which the function "name" is applied. There are several blocks,
    # and possibly useless variables; so the total dimension is num_blocks * block_dimension * (1+ uv_factor).
    functions = [
        ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
        for name in names
        for bd in [5, 30, 180]
        for uv_factor in [0, 5, 10]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                           1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]:  # , 10000]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def deceptive(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Very difficult objective functions: one is highly multimodal (infinitely many local optima),
    one has an infinite condition number, one has an infinitely long path towards the optimum.
    Looks somehow fractal."""
    seedg = create_seed_generator(seed)
    names = ["deceptivemultimodal", "deceptiveillcond", "deceptivepath"]
    optims = ["NGO", "Shiva", "DiagonalCMA", "PSO", "MiniQrDE", "MiniLhsDE", "MiniDE", "CMA", "QrDE", "DE", "LhsDE"]
    functions = [
        ArtificialFunction(name, block_dimension=2, num_blocks=n_blocks, rotation=rotation, aggregator=aggregator)
        for name in names
        for rotation in [False, True]
        for n_blocks in [1, 2, 8, 16]
        for aggregator in ["sum", "max"]
    ]
    for func in functions:
        for optim in optims:
            for budget in [25, 37, 50, 75, 87] + list(range(100, 20001, 500)):
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def largedoe(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Large scale experiments for one-shot optimizers.
    Very simple objective function (the sphere), various dimensions and numbers of useless variables."""
    seedg = create_seed_generator(seed)
    names = ["sphere"]
    optims = sorted(x for x, y in ng.optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x)
    functions = [
        ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
        for name in names
        for bd in [1, 4, 20]
        for uv_factor in [0, 10, 100]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def parallel(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Parallel optimization on 3 classical objective functions."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = ["ScrHammersleySearch", "NGO", "Shiva", "DiagonalCMA", "CMA", "PSO", "NaiveTBPSA", "OnePlusOne", "DE", "TwoPointsDE"]
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=int(budget / 5), seed=next(seedg))


@registry.register
def oneshot(seed: Optional[int] = None) -> Iterator[Experiment]:
    "One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar)"""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = sorted(x for x, y in ng.optimizers.registry.items() if y.one_shot)
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def multimodal(seed: Optional[int] = None, para: bool = False) -> Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal."""
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.
    optims = ["NGO", "Shiva", "DiagonalCMA", "NaiveTBPSA", "TBPSA",
              "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE", "OnePlusOne",
              "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE",
              "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS", "CM",
              "MultiCMA", "TripleCMA", "MultiScaleCMA"]
    if not para:
        optims += ["RSQP", "RCobyla", "RPowell", "SQPCMA", "SQP", "Cobyla", "Powell"]
    # + list(sorted(x for x, y in ng.optimizers.registry.items() if "chain" in x or "BO" in x))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000, 30000, 100000]:
                for nw in [1000] if para else [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def paramultimodal(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Parallel counterpart of the multimodal experiment."""
    internal_generator = multimodal(seed, para=True)
    for xp in internal_generator:
        yield xp


# pylint: disable=redefined-outer-name
@registry.register
def yabbob(seed: Optional[int] = None, parallel: bool = False, big: bool = False, noise: bool = False, hd: bool = False) -> Iterator[Experiment]:
    """Yet Another Black-Box Optimization Benchmark.
    Related to, but without special effort for exactly sticking to, the BBOB/COCO dataset.
    """
    seedg = create_seed_generator(seed)
    optims = ["NaiveTBPSA", "TBPSA", "NGO", "Shiva", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE", "OnePlusOne",
              "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE", "CMandAS2", "CMandAS"]
    if not parallel:
        optims += ["SQP", "Cobyla", "Powell", "chainCMASQP", "chainCMAPowell"]
    # optims += [x for x, y in ng.optimizers.registry.items() if "chain" in x]
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal", "bucherastrigin", "multipeak"]
    names += ["sphere", "doublelinearslope", "stepdoublelinearslope"]
    names += ["cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # Deceptive path is related to the sharp ridge function; there is a long path to the optimum.
    # Deceptive illcond is related to the difference of powers function; the conditioning varies as we get closer to the optimum.
    # Deceptive multimodal is related to the Weierstrass function and to the Schaffers function.
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=100 if noise else 0) for name in names
        for rotation in [True, False]
        for num_blocks in [1]
        for d in ([100, 1000, 3000] if hd else [2, 10, 50])
    ]
    for optim in optims:
        for function in functions:
            for budget in [50, 200, 800, 3200, 12800] if (not big and not noise) else [40000, 80000]:
                xp = Experiment(function, optim, num_workers=100 if parallel else 1,
                                budget=budget, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def yabigbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    internal_generator = yabbob(seed, parallel=False, big=True)
    for xp in internal_generator:
        yield xp


@registry.register
def yahdbbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    internal_generator = yabbob(seed, hd=True)
    for xp in internal_generator:
        yield xp


@registry.register
def yaparabbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Parallel optimization counterpart of yabbob."""
    internal_generator = yabbob(seed, parallel=True, big=False)
    for xp in internal_generator:
        yield xp


@registry.register
def yanoisybbob(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Noisy optimization counterpart of yabbob.
    This is supposed to be consistent with normal practices in noisy
    optimization: we distinguish recommendations and exploration.
    This is different from the original BBOB/COCO from that point of view.
    """
    internal_generator = yabbob(seed, noise=True)
    for xp in internal_generator:
        yield xp


@registry.register
def illcondi(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Testing optimizers on ill cond problems.
    """
    seedg = create_seed_generator(seed)
    optims = ["NGO", "Shiva", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE", "OnePlusOne", "SQP", "Cobyla",
              "Powell", "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE"]
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation in [True, False]
    ]
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def illcondipara(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Testing optimizers on ill-conditionned parallel optimization.
    """
    seedg = create_seed_generator(seed)
    optims = ["NGO", "Shiva", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE", "OnePlusOne",
              "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE",
              "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS", "CM",
              "MultiCMA", "TripleCMA", "MultiScaleCMA", "RSQP", "RCobyla", "RPowell", "SQPCMA"]
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation in [True, False]
    ]
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


def _positive_sum(data: np.ndarray) -> bool:
    if not isinstance(data, np.ndarray):
        raise ValueError(f"Unexpected inputs as np.ndarray, got {data}")
    return float(np.sum(data)) > 0


@registry.register
def constrained_illconditioned_parallel(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Many optimizers on ill cond problems with constraints.
    """
    seedg = create_seed_generator(seed)
    optims = ["NGO", "Shiva", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE", "OnePlusOne", "SQP", "Cobyla", "Powell",
              "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE",
              "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS", "CM",
              "MultiCMA", "TripleCMA", "MultiScaleCMA", "RSQP", "RCobyla", "RPowell", "SQPCMA"]
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation in [True, False]
    ]
    for func in functions:
        func.parametrization.register_cheap_constraint(_positive_sum)
    for optim in optims:
        for function in functions:
            for budget in [400, 4000, 40000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def doe_dim10(seed: Optional[int] = None) -> Iterator[Experiment]:
    """One-shot optimization in dimension 10 of the sphere function. No useless variables."""
    names = ["sphere"]
    seedg = create_seed_generator(seed)
    optims = sorted(x for x, y in ng.optimizers.registry.items() if y.one_shot and "arg" not in x and "mal" not in x)
    functions = [
        ArtificialFunction(name, block_dimension=bd, num_blocks=n_blocks, useless_variables=bd * uv_factor * n_blocks)
        for name in names
        for bd in [10]
        for uv_factor in [0]
        for n_blocks in [1]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def noisy(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    """
    seedg = create_seed_generator(seed)
    optims = ["NGO", "Shiva", "DiagonalCMA"] + sorted(
        x for x, y in ng.optimizers.registry.items() if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    )
    for budget in [50000]:
        for optim in optims:
            for d in [2, 20, 200]:
                for name in ["sphere", "rosenbrock"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(
                            name=name,
                            rotation=True,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def hdbo4d(seed: Optional[int] = None) -> Iterator[Experiment]:
    """All Bayesian optimization methods on various functions.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "BO" in x):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name, rotation=rotation, block_dimension=d, useless_variables=d * u, translation_factor=1.0
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def spsa_benchmark(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Some optimizers on a noisy optimization problem. This benchmark is based on the noise benchmark.
    """
    seedg = create_seed_generator(seed)
    optims = sorted(x for x, y in ng.optimizers.registry.items() if (any(e in x for e in "TBPSA SPSA".split()) and "iscr" not in x))
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=20, noise_level=10)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def realworld(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Realworld optimization. This experiment contains:
     - a subset of MLDA (excluding the perceptron: 10 functions rescaled or not.
     - ARCoating https://arxiv.org/abs/1904.02907: 1 function.
     - The 007 game: 1 function, noisy.
     - PowerSystem: a power system simulation problem.
     - STSP: a simple TSP problem.
     MLDA stuff, except the Perceptron.
    """
    funcs: List[Union[ExperimentFunction, rl.agents.TorchAgentFunction]] = [
        _mlda.Clustering.from_mlda(name, num, rescale) for name, num in [("Ruspini", 5), ("German towns", 10)] for rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]

    # Adding ARCoating.
    funcs += [ARCoating()]
    funcs += [PowerSystem(), PowerSystem(13)]
    funcs += [STSP(), STSP(500)]
    funcs += [game.Game("war")]
    funcs += [game.Game("batawaf")]
    funcs += [game.Game("flip")]
    funcs += [game.Game("guesswho")]
    funcs += [game.Game("bigguesswho")]

    # 007 with 100 repetitions, both mono and multi architectures.
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {'mono': rl.agents.Perceptron, 'multi': rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    runner = rl.EnvironmentRunner(env.copy(), num_repetitions=100, max_step=50)
    for archi in ["mono", "multi"]:
        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
        funcs += [func]
    seedg = create_seed_generator(seed)
    algos = ["NaiveTBPSA", "LargeScrHammersleySearch", "ScrHammersleySearch", "PSO", "OnePlusOne",
             "NGO", "Shiva", "DiagonalCMA", "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom", "RandomSearch", "HaltonSearch",
             "RandomScaleRandomSearch", "MiniDE"]
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in algos:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def simpletsp(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Simple TSP problems. Please note that the methods we use could be applied or complex variants, whereas
    specialized methods can not always do it; therefore this comparisons from a black-box point of view makes sense
    even if white-box methods are not included though they could do this more efficiently."""
    funcs = [STSP(10), STSP(100), STSP(1000), STSP(10000)]
    seedg = create_seed_generator(seed)
    algos = ["NaiveTBPSA", "SQP", "Powell", "LargeScrHammersleySearch", "ScrHammersleySearch", "PSO", "OnePlusOne",
             "NGO", "Shiva", "DiagonalCMA", "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom", "RandomSearch", "HaltonSearch",
             "RandomScaleRandomSearch", "MiniDE"]
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in algos:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def fastgames(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Optimization of policies for games, namely direct policy search."""
    funcs: List[ExperimentFunction] = []
    funcs += [game.Game("war")]
    funcs += [game.Game("batawaf")]
    funcs += [game.Game("flip")]
    funcs += [game.Game("guesswho")]
    funcs += [game.Game("bigguesswho")]
    seedg = create_seed_generator(seed)
    algos = ["NaiveTBPSA", "ScrHammersleySearch", "PSO", "OnePlusOne",
             "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom", "RandomSearch", "HaltonSearch",
             "RandomScaleRandomSearch", "MiniDE", "SplitOptimizer5", "NGO", "Shiva", "DiagonalCMA"]
    for budget in [1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in algos:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def powersystems(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Unit commitment problem, i.e. management of dams for hydroelectric planning."""
    funcs: List[ExperimentFunction] = []
    funcs += [PowerSystem(3)]
    funcs += [PowerSystem(num_dams=3, depth=5, width=5)]
    funcs += [PowerSystem(num_dams=3, depth=9, width=9)]
    funcs += [PowerSystem(5)]
    funcs += [PowerSystem(num_dams=5, depth=5, width=5)]
    funcs += [PowerSystem(num_dams=5, depth=9, width=9)]
    funcs += [PowerSystem(9)]
    funcs += [PowerSystem(num_dams=9, width=5, depth=5)]
    funcs += [PowerSystem(num_dams=9, width=9, depth=9)]
    funcs += [PowerSystem(13)]
    funcs += [PowerSystem(num_dams=13, width=5, depth=5)]
    funcs += [PowerSystem(num_dams=13, width=9, depth=9)]

    seedg = create_seed_generator(seed)
    algos = ["NaiveTBPSA", "ScrHammersleySearch", "PSO", "OnePlusOne",
             "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom", "RandomSearch", "HaltonSearch",
             "RandomScaleRandomSearch", "MiniDE", "SplitOptimizer5", "SplitOptimizer9", "SplitOptimizer",
             "NGO", "Shiva", "DiagonalCMA", "SplitOptimizer3", "SplitOptimizer13"]
    for budget in [1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in algos:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def powersystemssplit(seed: Optional[int] = None) -> Iterator[Experiment]:
    """Unit commitment problem, i.e. management of dams for hydroelectric planning.
    Bigger budget than the powersystems problem."""
    funcs: List[ExperimentFunction] = []
    funcs += [PowerSystem(3)]
    funcs += [PowerSystem(num_dams=3, depth=5, width=5)]
    funcs += [PowerSystem(num_dams=3, depth=9, width=9)]
    funcs += [PowerSystem(5)]
    funcs += [PowerSystem(num_dams=5, depth=5, width=5)]
    funcs += [PowerSystem(num_dams=5, depth=9, width=9)]
    funcs += [PowerSystem(9)]
    funcs += [PowerSystem(num_dams=9, width=5, depth=5)]
    funcs += [PowerSystem(num_dams=9, width=9, depth=9)]
    funcs += [PowerSystem(13)]
    funcs += [PowerSystem(num_dams=13, width=5, depth=5)]
    funcs += [PowerSystem(num_dams=13, width=9, depth=9)]

    seedg = create_seed_generator(seed)
    algos = ["NGO", "Shiva", "DiagonalCMA",
             "CMA", "Zero", "RandomSearch",
             "DE", "PSO", "SplitOptimizer5", "SplitOptimizer9", "SplitOptimizer", "SplitOptimizer3", "SplitOptimizer13"]
    for budget in [25600, 51200, 102400, 204800, 409600]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in algos:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mlda(seed: Optional[int] = None) -> Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed."""
    funcs: List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale) for name, num in [("Ruspini", 5), ("German towns", 10)] for rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Perceptron.from_mlda(name) for name in ["quadratic", "sine", "abs", "heaviside"]]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]
    seedg = create_seed_generator(seed)
    algos = ["NaiveTBPSA", "LargeScrHammersleySearch", "ScrHammersleySearch", "PSO", "OnePlusOne",
             "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom", "RandomSearch", "HaltonSearch",
             "RandomScaleRandomSearch", "MiniDE", "NGO", "Shiva", "DiagonalCMA"]
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in algos:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


# @registry.register
# def mldaas(seed: Optional[int] = None) -> Iterator[Experiment]:
#     funcs: List[ExperimentFunction] = [
#         _mlda.Clustering.from_mlda(name, num, rescale) for name, num in [("Ruspini", 5), ("German towns", 10)] for rescale in [True, False]
#     ]
#     funcs += [
#         _mlda.SammonMapping.from_mlda("Virus", rescale=False),
#         _mlda.SammonMapping.from_mlda("Virus", rescale=True),
#         _mlda.SammonMapping.from_mlda("Employees"),
#     ]
#     funcs += [_mlda.Perceptron.from_mlda(name) for name in ["quadratic", "sine", "abs", "heaviside"]]
#     funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]
#     seedg = create_seed_generator(seed)
#     algos = ["NaiveTBPSA", "ScrHammersleySearch", "PSO", "OnePlusOne", "NGO", "Shiva", "DiagonalCMA", "CMA", "OnePointDE", "TwoPointsDE", "QrDE", "LhsDE",
#              "Zero", "PortfolioDiscreteOnePlusOne", "CauchyOnePlusOne", "RandomSearch", "RandomSearchPlusMiddlePoint",
#              "HaltonSearchPlusMiddlePoint", "MiniQrDE", "HaltonSearch", "RandomScaleRandomSearch", "MiniDE", "DiscreteOnePlusOne",
#              "ScrHaltonSearch", "ScrHammersleySearchPlusMiddlePoint", "HaltonSearch", "MilliCMA", "MicroCMA"]
#     # pylint: disable=too-many-nested-blocks
#     algos += ["Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS",
#               "CM", "MultiCMA", "TripleCMA", "MultiScaleCMA"]
#     for budget in [9600, 12800, 25600]:  # , 51200]:#, 102400]:
#         for num_workers in [10, 100, 1000]:  # [1, 10, 100]:
#             for algo in algos:
#                 for func in funcs:
#                     if num_workers < budget:
#                         xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
#                         if not xp.is_incoherent:
#                             yield xp


@registry.register
def arcoating(seed: Optional[int] = None) -> Iterator[Experiment]:
    """AR coating. Problems about optical properties of nanolayers."""
    seedg = create_seed_generator(seed)
    algos = ["NaiveTBPSA", "Cobyla", "SQP", "Powell", "LargeScrHammersleySearch", "ScrHammersleySearch", "PSO",
             "OnePlusOne", "NGO", "Shiva", "DiagonalCMA", "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom"]
    # for budget in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    for budget in [100 * 5 ** k for k in range(6)]:  # from 100 to 312500
        for num_workers in [1, 10, 100]:
            for algo in algos:
                for func in [ARCoating(10, 400), ARCoating(35, 700), ARCoating(70, 1000)]:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def double_o_seven(seed: Optional[int] = None) -> Iterator[Experiment]:
    # pylint: disable=too-many-locals
    seedg = create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {'mono': rl.agents.Perceptron, 'multi': rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    for num_repetitions in [1, 10, 100]:
        for archi in ["mono", "multi"]:
            dde = ng.optimizers.DifferentialEvolution(crossover="dimension").with_name("DiscreteDE")
            for optim in ["PSO", "NGO", "Shiva", "DiagonalCMA", "CMA", "DE", "TwoPointsDE", "TBPSA", "OnePlusOne", "Zero",
                          "RandomSearch", "AlmostRotationInvariantDE", dde]:
                for env_budget in [5000, 10000, 20000, 40000]:
                    for num_workers in [1, 10, 100]:
                        # careful, not threadsafe
                        runner = rl.EnvironmentRunner(env.copy(), num_repetitions=num_repetitions, max_step=50)
                        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
                        opt_budget = env_budget // num_repetitions
                        yield Experiment(func, optim, budget=opt_budget, num_workers=num_workers, seed=next(seedg))  # type: ignore


# Intermediate definition for building a multiobjective problem.
class PackedFunctions(ExperimentFunction):

    def __init__(self, functions: List[ArtificialFunction], upper_bounds: np.ndarray) -> None:
        self._functions = functions
        self._upper_bounds = upper_bounds
        self.multiobjective = MultiobjectiveFunction(self._mo, upper_bounds)
        super().__init__(self.multiobjective, self._functions[0].parametrization)
        # TODO add descriptors?

    def _mo(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.array([f(*args, **kwargs) for f in self._functions])

    def copy(self) -> "PackedFunctions":
        return PackedFunctions([f.copy() for f in self._functions], self._upper_bounds)


@registry.register
def multiobjective_example(seed: Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optims = ["NaiveTBPSA", "PSO", "DE", "LhsDE", "RandomSearch", "NGO", "Shiva", "DiagonalCMA", "CMA", "OnePlusOne", "TwoPointsDE"]
    mofuncs: List[PackedFunctions] = []
    for name1 in ["sphere", "cigar"]:
        for name2 in ["sphere", "cigar", "hm"]:
            mofuncs += [PackedFunctions([ArtificialFunction(name1, block_dimension=7),
                                         ArtificialFunction(name2, block_dimension=7)],
                                        upper_bounds=np.array((50., 50.)))]
            for name3 in ["sphere", "ellipsoid"]:
                mofuncs += [PackedFunctions([ArtificialFunction(name1, block_dimension=6),
                                             ArtificialFunction(name3, block_dimension=6),
                                             ArtificialFunction(name2, block_dimension=6)],
                                            upper_bounds=np.array((100, 100, 1000.)))]
    for mofunc in mofuncs:
        for optim in optims:
            for budget in list(range(100, 2901, 400)):
                for nw in [1, 100]:
                    yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def manyobjective_example(seed: Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optims = ["NaiveTBPSA", "PSO", "DE", "LhsDE", "RandomSearch", "NGO", "Shiva", "DiagonalCMA", "CMA", "OnePlusOne", "TwoPointsDE"]
    mofuncs: List[PackedFunctions] = []
    for name1 in ["sphere", "cigar"]:
        for name2 in ["sphere", "hm"]:
            for name3 in ["sphere", "ellipsoid"]:
                for name4 in ["rastrigin", "rosenbrock"]:
                    for name5 in ["hm", "rosenbrock"]:
                        for name6 in ["rastrigin", "cigar"]:
                            mofuncs += [PackedFunctions([ArtificialFunction(name1, block_dimension=6),
                                                         ArtificialFunction(name2, block_dimension=6),
                                                         ArtificialFunction(name3, block_dimension=6),
                                                         ArtificialFunction(name4, block_dimension=6),
                                                         ArtificialFunction(name5, block_dimension=6),
                                                         ArtificialFunction(name6, block_dimension=6)],
                                                        upper_bounds=np.array((100, 100, 1000., 7., 300., 500.)))]
    for mofunc in mofuncs:
        for optim in optims:
            for budget in list(range(100, 5901, 400)):
                for nw in [1, 100]:
                    yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def far_optimum_es(seed: tp.Optional[int] = None) -> Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    popsizes = [5, 40]
    es = [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=False, popsize=pop)
          for recomb in [0, 1] for pop in popsizes]
    es += [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=only, popsize=pop,
                                         offsprings=10 if pop == 5 else 60)
           for only in [True, False] for recomb in [0, 1] for pop in popsizes]
    optimizers = ["CMA", "TwoPointsDE"] + es  # type: ignore
    for func in FarOptimumFunction.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def photonics(seed: tp.Optional[int] = None) -> Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    popsizes = [10, 40, 100]
    es = [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=False, popsize=pop)
          for recomb in [0.1, 1] for pop in popsizes]
    es += [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=only, popsize=pop,
                                         offsprings={10: 20, 40: 60, 100: 150}[pop])
           for only in [True, False] for recomb in [0.1, 1] for pop in popsizes]
    algos = ["TwoPointsDE", "DE", "PSO", "OnePlusOne", "ParametrizationDE", "NaiveTBPSA"] + es  # type: ignore
    for method in ["clipping", "tanh", "arctan"]:
        # , "chirped"]]:  # , "morpho"]]:
        for func in [Photonics(x, 60 if x == "morpho" else 80, bounding_method=method) for x in ["bragg"]]:
            for budget in [1e2, 1e3, 1e4, 1e5, 1e6]:
                for algo in algos:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp
