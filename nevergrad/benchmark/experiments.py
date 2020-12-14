# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import typing as tp
import itertools
import numpy as np
import nevergrad as ng
from nevergrad.optimization.base import ConfiguredOptimizer
import nevergrad.functions.corefuncs as corefuncs
from nevergrad.functions import base as fbase
from nevergrad.functions import ExperimentFunction
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import FarOptimumFunction
from nevergrad.functions import PBT
from nevergrad.functions.ml import MLTuning
from nevergrad.functions import mlda as _mlda
from nevergrad.functions.photonics import Photonics
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions import images as imagesxp
from nevergrad.functions.powersystems import PowerSystem
from nevergrad.functions.stsp import STSP
from nevergrad.functions.rocket import Rocket
from nevergrad.functions.mixsimulator import OptimizeMix
from nevergrad.functions import control
from nevergrad.functions import rl
from nevergrad.functions.games import game
from nevergrad.functions import iohprofiler
from .xpbase import Experiment as Experiment
from .xpbase import create_seed_generator
from .xpbase import registry as registry  # noqa
from .optgroups import get_optimizers

# register all frozen experiments
from . import frozenexperiments  # noqa # pylint: disable=unused-import

# pylint: disable=stop-iteration-return, too-many-nested-blocks, too-many-locals, line-too-long
# pylint: disable=too-many-lines
# for black (since lists are way too long...):
# fmt: off


class _Constraint:

    def __init__(self, name: str, as_bool: bool) -> None:
         self.name = name
         self.as_bool = as_bool

    def __call__(self, data: np.ndarray) -> tp.Union[bool, float]:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Unexpected inputs as np.ndarray, got {data}")
        if self.name == "sum":
            value = float(np.sum(data))
        elif self.name == "diff":
            value = float(np.sum(data[::2]) - np.sum(data[1::2]))
        elif self.name == "second_diff":
            value = float(2 * np.sum(data[1::2]) - 3 * np.sum(data[::2]))
        elif self.name == "ball":
            value = float(np.sum(np.square(data))) - float(len(data)) - float(np.sqrt(len(data)))  # Most points violate the constraint.
        else:
            raise NotImplementedError(f"Unknown function {self.name}")
        return value > 0 if self.as_bool else value


default_optims: tp.Optional[tp.List[str]] = None  # ["NGO10", "CMA", "Shiwa"]


def keras_tuning(seed: tp.Optional[int] = None, overfitter: bool = False, seq: bool = False) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on scikit models."""
    seedg = create_seed_generator(seed)
    # Continuous case,

    # First, a few functions with constraints.
    optims = ["PSO", "OnePlusOne"]
    if default_optims is not None:
        optims = default_optims
    for dimension in [None]:
        for dataset in (
                ["kerasBoston", "diabetes", "auto-mpg", "red-wine", "white-wine"]):
            function = MLTuning(regressor="keras_dense_nn", data_dimension=dimension, dataset=dataset,
                                overfitter=overfitter)
            for budget in [50, 150, 500]:
                for num_workers in [1, budget // 4] if seq else [budget]:  # Seq for sequential optimization experiments.
                    for optim in optims:
                        xp = Experiment(function, optim, num_workers=num_workers,
                                        budget=budget, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


def mltuning(seed: tp.Optional[int] = None, overfitter: bool = False, seq: bool = False) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on scikit models."""
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore

    if default_optims is not None:
        optims = default_optims
    for dimension in [None, 1, 2, 3]:
        for regressor in ["mlp", "decision_tree", "decision_tree_depth"]:
            for dataset in (
                    ["boston", "diabetes", "auto-mpg", "red-wine", "white-wine"] if dimension is None else ["artificialcos", "artificial",
                                                                      "artificialsquare"]):
                function = MLTuning(regressor=regressor, data_dimension=dimension, dataset=dataset,
                                    overfitter=overfitter)
                for budget in [50, 150, 500]:
                    for num_workers in [1, budget // 4] if seq else [budget]:  # Seq for sequential optimization experiments.
                        for optim in optims:
                            xp = Experiment(function, optim, num_workers=num_workers,
                                            budget=budget, seed=next(seedg))
                            if not xp.is_incoherent:
                                yield xp


def naivemltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True)


# We register only the sequential counterparts for the moment.
@registry.register
def seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of keras tuning."""
    return keras_tuning(seed, overfitter=False, seq=True)


# We register only the sequential counterparts for the moment.
@registry.register
def naive_seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of mltuning."""
    return keras_tuning(seed, overfitter=True, seq=True)


# We register only the sequential counterparts for the moment.
@registry.register
def seqmltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of mltuning."""
    return mltuning(seed, overfitter=False, seq=True)


@registry.register
def naiveseqmltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True, seq=True)


# pylint:disable=too-many-branches
@registry.register
def yawidebbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Yet Another Wide Black-Box Optimization Benchmark.
    The goal is basically to have a very wide family of problems: continuous and discrete,
    noisy and noise-free, mono- and multi-objective,  constrained and not constrained, sequential
    and parallel.

    TODO(oteytaud): this requires a significant improvement, covering mixed problems and different types of constraints.
    """
    seedg = create_seed_generator(seed)
    # Continuous case

    # First, a few functions with constraints.
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation
        in [True, False]
    ]
    for func in functions:
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=False))

    # Then, let us build a constraint-free case. We include the noisy case.
    names = ["hm", "rastrigin", "sphere", "doublelinearslope", "ellipsoid"]

    # names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    functions += [
        ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=nl, split=split)
        for name in names              # period 5
        for rotation in [True, False]  # period 2
        for nl in [0., 100.]           # period 2
        for num_blocks in [1, 8]       # period 2
        for d in [2, 40, 3000, 10000]  # period 4
        for split in [True, False]     # period 2
    ][::13]  # 13 is coprime with all periods above so we sample correctly the possibilities.
    # This problem is intended as a stable basis forever.
    # The list of optimizers should contain only the basic for comparison and "baselines".
    optims: tp.List[str] = ["NGOpt8"] + get_optimizers("baselines", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims
    for optim in optims:
        for function in functions:
            for budget in [50, 1500, 50000]:
                for nw in [1, budget] + ([] if budget <= 300 else [300]):
                    xp = Experiment(function, optim, num_workers=nw,
                                    budget=budget, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp
    # Discrete, unordered.
    for nv in [200, 2000]:
        for arity in [2, 7, 37]:
            instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
            for name in ["onemax", "leadingones", "jump"]:
                dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name("transition"))
                dfunc.add_descriptors(arity=arity)
                for optim in optims:
                    for budget in [500, 5000]:
                        for nw in [1, 100]:
                            yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))

    # The multiobjective case.
    # TODO the upper bounds are really not well set for this experiment with cigar
    mofuncs: tp.List[fbase.MultiExperiment] = []
    for name1 in ["sphere", "ellipsoid"]:
        for name2 in ["sphere", "hm"]:
            mofuncs += [fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=7),
                                               ArtificialFunction(name2, block_dimension=7)],
                                              upper_bounds=np.array((100., 100.)))]
    for name1 in ["sphere", "ellipsoid"]:
        for name2 in ["sphere", "hm"]:
            for name3 in ["sphere", "hm"]:
                mofuncs += [fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=7),
                                                   ArtificialFunction(name2, block_dimension=7),
                                                   ArtificialFunction(name3, block_dimension=7)],
                                                  upper_bounds=np.array((100., 100., 100.)))]
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [2000, 8000]:
                for nw in [1, 100]:
                    yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))


# pylint: disable=redefined-outer-name
@registry.register
def parallel_small_budget(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization with small budgets
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "multipeak"]
    names += ["sphere", "cigar", "ellipsoid", "altellipsoid"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # funcs
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation)
        for name in names
        for rotation in [True, False]
        for d in [2, 4, 8]
    ]
    budgets = [10, 50, 100, 200, 400]
    for optim in optims:
        for function in functions:
            for budget in budgets:
                for nw in [2, 8, 16]:
                    for batch in [True, False]:
                        if nw < budget / 4:
                            xp = Experiment(function, optim, num_workers=nw, budget=budget, batch_mode=batch,
                                            seed=next(seedg))
                            if not xp.is_incoherent:
                                yield xp


@registry.register
def instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Comparison of optimization algorithms equipped with distinct instrumentations.
    Onemax, Leadingones, Jump function."""
    # Discrete, unordered.

    optims = ["DiscreteOnePlusOne", "Shiwa", "CMA", "PSO", "TwoPointsDE", "DE", "OnePlusOne",
              "AdaptiveDiscreteOnePlusOne",
              "CMandAS2", "PortfolioDiscreteOnePlusOne", "DoubleFastGADiscreteOnePlusOne", "MultiDiscrete",
              "DiscreteBSOOnePlusOne", "AnisotropicAdaptiveDiscreteOnePlusOne", "DiscreteLenglerOnePlusOne"]

    if default_optims is not None:
        optims = default_optims
    seedg = create_seed_generator(seed)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax"]:
                if instrum_str == "Softmax":
                    instrum: ng.p.Parameter = ng.p.Choice(range(arity), repetitions=nv)
                    # Equivalent to, but much faster than, the following:
                    # instrum = ng.p.Tuple(*(ng.p.Choice(range(arity)) for _ in range(nv)))
#                 else:
#                     assert instrum_str == "Threshold"
#                     # instrum = ng.p.Tuple(*(ng.p.TransitionChoice(range(arity)) for _ in range(nv)))
#                     init = np.random.RandomState(seed=next(seedg)).uniform(-0.5, arity -0.5, size=nv)
#                     instrum = ng.p.Array(init=init).set_bounds(-0.5, arity -0.5)  # type: ignore
                else:
                    assert instrum_str == "Unordered"
                    instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    dfunc.add_descriptors(arity=arity)
                    for optim in optims:
                        for nw in [1, 10]:
                            for budget in [50, 500, 5000]:
                                yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))


@registry.register
def sequential_instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of instrum_discrete."""
    # Discrete, unordered.
    optims = ["DiscreteOnePlusOne", "DiscreteDoerrOnePlusOne", "DiscreteLenglerOnePlusOne", "FastGADiscreteOnePlusOne",
              "DiscreteBSOOnePlusOne", "PortfolioDiscreteOnePlusOne", "DoubleFastGADiscreteOnePlusOne",
              "AnisotropicAdaptiveDiscreteOnePlusOne"]

    seedg = create_seed_generator(seed)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered"]:
                assert instrum_str == "Unordered"
                instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str))
                    dfunc.add_descriptors(arity=arity)
                    for optim in optims:
                        for budget in [50, 500, 5000, 50000]:
                            yield Experiment(dfunc, optim, budget=budget, seed=next(seedg))


@registry.register
def deceptive(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Very difficult objective functions: one is highly multimodal (infinitely many local optima),
    one has an infinite condition number, one has an infinitely long path towards the optimum.
    Looks somehow fractal."""
    seedg = create_seed_generator(seed)
    names = ["deceptivemultimodal", "deceptiveillcond", "deceptivepath"]
    optims = ["NGO", "Shiwa", "DiagonalCMA", "PSO", "MiniQrDE", "MiniLhsDE", "MiniDE", "CMA", "QrDE", "DE", "LhsDE"]
    if default_optims is not None:
        optims = default_optims
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
def parallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization on 3 classical objective functions: sphere, rastrigin, cigar.
    The number of workers is 20 % of the budget.
    Testing both no useless variables and 5/6 of useless variables."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims: tp.List[str] = get_optimizers("parallel_basics", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims
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
def harderparallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization on 4 classical objective functions. More distinct settings than << parallel >>."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar", "ellipsoid"]
    optims = ["NGOpt8"] + get_optimizers("emna_variants", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [5, 25]
        for uv_factor in [0, 5]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                for num_workers in [int(budget / 10), int(budget / 5), int(budget / 3)]:
                    yield Experiment(func, optim, budget=budget, num_workers=num_workers, seed=next(seedg))


@registry.register
def oneshot(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    0 or 5 dummy variables per real variable.
    Base dimension 3 or 25.
    budget 30, 100 or 3000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = sorted(x for x, y in ng.optimizers.registry.items() if y.one_shot and "4" not in x and "7" not in x and
                    "LHS" not in x and "alton" not in x
                    and ("ando" not in x or "QO" in x))  # QORandomSearch is the only valid variant of RandomSearch.
    if default_optims is not None:
        optims = default_optims
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
def doe(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Base dimension 2000 or 20000. No rotation, no dummy variable.
    Budget 30, 100, 3000, 10000, 30000, 100000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = sorted(x for x, y in ng.optimizers.registry.items() if y.one_shot)
    if default_optims is not None:
        optims = default_optims
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [2000, 20000]  # 3, 10, 25, 200, 2000]
        for uv_factor in [0]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000, 30000, 100000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def newdoe(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Tested on more dimensionalities than doe, namely 20, 200, 2000, 20000. No dummy variables.
    Budgets 30, 100, 3000, 10000, 30000, 100000, 300000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = sorted(
        x for x, y in ng.optimizers.registry.items() if y.one_shot and "hiva" not in str(y) and "NGO" not in str(
            y) and ("ando" in x or "HCH" in x or "LHS" in x or "eta" in x) and "mmers" not in x and "alto" not in x)
    if default_optims is not None:
        optims = default_optims
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [2000, 20, 200, 20000]  # 3, 10, 25, 200, 2000]
        for uv_factor in [0]
    ]
    budgets = [30, 100, 3000, 10000, 30000, 100000, 300000]
    for func in functions:
        for optim in optims:
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def fiveshots(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Five-shots optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    Base dimension 3 or 25. 0 or 5 dummy variable per real variable. Budget 30, 100 or 3000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = sorted(x for x, y in ng.optimizers.registry.items() if y.one_shot)
    optims += ["CMA", "NGOpt8", "DE"]
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    if default_optims is not None:
        optims = default_optims
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget // 5, seed=next(seedg))


@registry.register
def multimodal(seed: tp.Optional[int] = None, para: bool = False) -> tp.Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal.
    0 or 5 dummy variable per real variable.
    Base dimension 3 or 25.
    Budget in 3000, 10000, 30000, 100000.
    Sequential.
    """
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.
    optims = ["NGOpt8", "CMandAS", "DiagonalCMA", "CMA", "TripleCMA", "CMandAS2", "PSO"]
    if default_optims is not None:
        optims = default_optims
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
def hdmultimodal(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal. Similar to multimodal, but dimension 20 or 100 or 1000. Budget 1000 or 10000, sequential."""
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.
    optims = ["NGOpt8", "MultiCMA", "RPowell", "CMA", "TripleCMA"]
    optims = ["NaiveTBPSA"]
    if default_optims is not None:
        optims = default_optims
    # + list(sorted(x for x, y in ng.optimizers.registry.items() if "chain" in x or "BO" in x))
    functions = [
        ArtificialFunction(name, block_dimension=bd)
        for name in names
        for bd in [1000, 6000, 36000]  # This has been modified, given that it was not sufficiently high-dimensional for its name.
    ]
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000]:
                for nw in [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def paramultimodal(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel counterpart of the multimodal experiment: 1000 workers."""
    return multimodal(seed, para=True)


# pylint: disable=redefined-outer-name,too-many-arguments
@registry.register
def yabbob(seed: tp.Optional[int] = None, parallel: bool = False, big: bool = False, small: bool = False,
           noise: bool = False, hd: bool = False, constraint_case: int = 0, split: bool = False) -> tp.Iterator[Experiment]:
    """Yet Another Black-Box Optimization Benchmark.
    Related to, but without special effort for exactly sticking to, the BBOB/COCO dataset.
    Dimension 2, 10 and 50.
    Budget 50, 200, 800, 3200, 12800.
    Both rotated or not rotated.
    """
    seedg = create_seed_generator(seed)

    # List of objective functions.
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal", "bucherastrigin",
             "multipeak"]
    names += ["sphere", "doublelinearslope", "stepdoublelinearslope"]
    names += ["cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # Deceptive path is related to the sharp ridge function; there is a long path to the optimum.
    # Deceptive illcond is related to the difference of powers function; the conditioning varies as we get closer to the optimum.
    # Deceptive multimodal is related to the Weierstrass function and to the Schaffers function.

    # Parametrizing the noise level.
    if noise:
        noise_level = 100000 if hd else 100
    else:
        noise_level = 0

    # Choosing the list of optimizers.
    optims: tp.List[str] = ["chainMetaModelSQP", "NGOpt8", "chainCMASQP", "MetaModel", "CMA", "CMandAS2"]
    if noise:
        optims += ["TBPSA", "SQP", "NoisyDiscreteOnePlusOne"]
    if hd:
        optims += ["OnePlusOne"]
        optims += get_optimizers("splitters", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims

    # List of objective functions.
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation, noise_level=noise_level, split=split) for name in names
        for rotation in [True, False]
        for num_blocks in ([1] if not split else [7, 12])
        for d in ([100, 1000, 3000] if hd else [2, 10, 50])
    ]

    # We possibly add constraints.
    max_num_constraints = 4
    constraints: tp.List[tp.Any] = [_Constraint(name, as_bool)
        for as_bool in [False, True]
        for name in ["sum", "diff", "second_diff", "ball"]
        ]
    assert constraint_case < len(constraints) + max_num_constraints, (
        "constraint_case should be in 0, 1, ..., {len(constraints) + max_num_constraints - 1} (0 = no constraint).")
    # We reduce the number of tests when there are constraints, as the number of cases
    # is already multiplied by the number of constraint_case.
    for func in functions[::13 if constraint_case > 0 else 1]:
        # We add a window of the list of constraints. This windows finishes at "constraints" (hence, is empty if
        # constraint_case=0).
        for constraint in constraints[max(0, constraint_case - max_num_constraints):constraint_case]:
            func.parametrization.register_cheap_constraint(constraint)

    budgets = [40000, 80000, 160000, 320000] if (big and not noise) else [50, 200, 800, 3200, 12800]
    if (small and not noise):
        budgets = [10, 20, 40]
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp = Experiment(function, optim, num_workers=100 if parallel else 1,
                                budget=budget, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def yaconstrainedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    cases = 8 # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=i) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yahdnoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True, noise=True)


@registry.register
def yabigbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, parallel=False, big=True)


@registry.register
def yasplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, parallel=False, split=True)


@registry.register
def yahdsplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, hd=True, split=True)


@registry.register
def yasmallbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget."""
    return yabbob(seed, parallel=False, big=False, small=True)


@registry.register
def yahdbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True)


@registry.register
def yaparabbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization counterpart of yabbob."""
    return yabbob(seed, parallel=True, big=False)


@registry.register
def yanoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization counterpart of yabbob.
    This is supposed to be consistent with normal practices in noisy
    optimization: we distinguish recommendations and exploration.
    This is different from the original BBOB/COCO from that point of view.
    """
    return yabbob(seed, noise=True)


@registry.register
def illcondi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on ill cond problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """
    seedg = create_seed_generator(seed)
    optims = ["NGOpt8", "Cobyla", "SQP", "OnePlusOne", "DiagonalCMA", "MiniDE"]
    if default_optims is not None:
        optims = default_optims
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation
        in [True, False]
    ]
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def illcondipara(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on ill-conditionned parallel optimization.
    50 workers in parallel.
    """
    seedg = create_seed_generator(seed)
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation
        in [True, False]
    ]
    optims = ["DiagonalCMA", "MetaModel", "NGOpt8", "CMA"]
    for function in functions:
        for budget in [100, 1000, 10000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=50, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def constrained_illconditioned_parallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Many optimizers on ill cond problems with constraints.
    """
    seedg = create_seed_generator(seed)
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation) for name in ["cigar", "ellipsoid"] for rotation
        in [True, False]
    ]
    for func in functions:
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=False))
    for function in functions:
        for budget in [400, 4000, 40000]:
            optims: tp.List[str] = get_optimizers("large", seed=next(seedg))  # type: ignore
            for optim in optims:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def ranknoisy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    Cigar, Altcigar, Ellipsoid, Altellipsoid.
    Dimension 200, 2000, 20000.
    Budget 25000, 50000, 100000.
    No rotation.
    Noise level 10.
    With or without noise dissymmetry.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [  # type: ignore
        "OptimisticNoisyOnePlusOne", "OptimisticDiscreteOnePlusOne"]
    if default_optims is not None:
        optims = default_optims
    # optims += ["NGO", "Shiwa", "DiagonalCMA"] + sorted(
    #    x for x, y in ng.optimizers.registry.items() if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    # )
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [20000, 200, 2000]:
                for name in ["cigar", "altcigar", "ellipsoid", "altellipsoid"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(
                            name=name,
                            rotation=False,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def noisy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    Sphere, Rosenbrock, Cigar, Hm (= highly multimodal).
    Noise level 10.
    Noise dyssymmetry or not.
    Dimension 2, 20, 200, 2000.
    Budget 25000, 50000, 100000.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [  # type: ignore
        "OptimisticNoisyOnePlusOne", "OptimisticDiscreteOnePlusOne"]
    optims += ["NGO", "Shiwa", "DiagonalCMA"] + sorted(
        x for x, y in ng.optimizers.registry.items() if
        ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    )
    if default_optims is not None:
        optims = default_optims
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [2, 20, 200, 2000]:
                for name in ["sphere", "rosenbrock", "cigar", "hm"]:
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
def paraalldes(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All DE methods on various functions. Parallel version.
    Dimension 5, 20, 100, 500, 2500.
    Sphere, Cigar, Hm, Ellipsoid.
    No rotation.
    """
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x and "Tune" in x):
            for rotation in [False]:
                for d in [5, 20, 100, 500, 2500]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name, rotation=rotation, block_dimension=d, useless_variables=d * u,
                                translation_factor=1.0
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg),
                                             num_workers=max(d, budget // 6))


@registry.register
def parahdbo4d(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All Bayesian optimization methods on various functions. Parallel version
    Dimension 20 and 2000.
    Budget 25, 31, 37, 43, 50, 60.
    Sphere, Cigar, Hm, Ellipsoid.
    No rotation.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "BO" in x and "Tune" in x):
            for rotation in [False]:
                for d in [20, 2000]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name, rotation=rotation, block_dimension=d, useless_variables=d * u,
                                translation_factor=1.0
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg),
                                             num_workers=max(d, budget // 6))


@registry.register
def alldes(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All DE methods on various functions.
    Dimension 5, 20, 100.
    Sphere, Cigar, Hm, Ellipsoid.
    Budget 10, 100, 1000, 10000, 100000.
    """
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in default_optims if default_optims else sorted(
                x for x, y in ng.optimizers.registry.items() if "DE" in x or "Shiwa" in x):
            for rotation in [False]:
                for d in [5, 20, 100]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name, rotation=rotation, block_dimension=d, useless_variables=d * u,
                                translation_factor=1.0
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def hdbo4d(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All Bayesian optimization methods on various functions.
    Budget 25, 31, 37, 43, 50, 60.
    Dimension 20.
    Sphere, Cigar, Hm, Ellipsoid.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in get_optimizers("all_bo"):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name, rotation=rotation, block_dimension=d, useless_variables=d * u,
                                translation_factor=1.0
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def spsa_benchmark(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Some optimizers on a noisy optimization problem. This benchmark is based on the noisy benchmark.
    Budget 500, 1000, 2000, 4000, ... doubling... 128000.
    Rotation or not.
    Sphere, Sphere4, Cigar.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("spsa")  # type: ignore
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ArtificialFunction(name=name, rotation=rotation, block_dimension=20, noise_level=10)
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def realworld(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Realworld optimization. This experiment contains:

     - a subset of MLDA (excluding the perceptron: 10 functions rescaled or not.
     - ARCoating https://arxiv.org/abs/1904.02907: 1 function.
     - The 007 game: 1 function, noisy.
     - PowerSystem: a power system simulation problem.
     - STSP: a simple TSP problem.
     -  MLDA, except the Perceptron.

    Budget 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800.
    Sequential or 10-parallel or 100-parallel.
    """
    funcs: tp.List[tp.Union[ExperimentFunction, rl.agents.TorchAgentFunction]] = [
        _mlda.Clustering.from_mlda(name, num, rescale) for name, num in [("Ruspini", 5), ("German towns", 10)] for
        rescale in [True, False]
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
    optims = ["NGOpt8", "CMA", "DE"]
    if default_optims is not None:
        optims = default_optims
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def rocket(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Rocket simulator. Maximize max altitude by choosing the thrust schedule, given a total thrust.
    Budget 25, 50, ..., 1600.
    Sequential or 30 workers."""
    funcs = [Rocket()]
    seedg = create_seed_generator(seed)
    optims = ["DiagonalCMA", "MetaModel", "CMA", "DE", "NGOpt8"]
    if default_optims is not None:
        optims = default_optims
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mixsimulator(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MixSimulator of power plants
    Budget 20, 40, ..., 1600.
    Sequential or 30 workers."""
    funcs = [OptimizeMix()]
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims
    seq = np.arange(0, 1601, 20)
    for budget in seq:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def control_problem(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MuJoCo testbed. Learn linear policy for different control problems.
    Budget 500, 1000, 3000, 5000."""
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [control.Swimmer(num_rollouts=num_rollouts, random_state=seed),
             control.HalfCheetah(num_rollouts=num_rollouts, random_state=seed),
             control.Hopper(num_rollouts=num_rollouts, random_state=seed),
             control.Walker2d(num_rollouts=num_rollouts, random_state=seed),
             control.Ant(num_rollouts=num_rollouts, random_state=seed),
             control.Humanoid(num_rollouts=num_rollouts, random_state=seed)
             ]

    sigmas = [0.1, 0.1, 0.1, 0.1, 0.01, 0.001]
    funcs2 = []
    for sigma, func in zip(sigmas, funcs):
        f = func.copy()
        param: ng.p.Array = f.parametrization.copy()  # type: ignore
        param.set_mutation(sigma=sigma).set_name(f"sigma={sigma}")
        f.parametrization = param
        f.parametrization.freeze()
        funcs2.append(f)

    optims = ["CMA", "DiagonalCMA", "NGOpt8"]

    for budget in [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000, 3000, 5000, 8000, 16000, 32000, 64000]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs2:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def simpletsp(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Simple TSP problems. Please note that the methods we use could be applied or complex variants, whereas
    specialized methods can not always do it; therefore this comparisons from a black-box point of view makes sense
    even if white-box methods are not included though they could do this more efficiently.
    10, 100, 1000, 10000 cities.
    Budgets doubling from 25, 50, 100, 200, ... up  to 25600

    """
    funcs = [STSP(10), STSP(100), STSP(1000), STSP(10000)]
    seedg = create_seed_generator(seed)
    optims = ["PortfolioDiscreteOnePlusOne", "NGOpt8", "OnePlusOne", "DiagonalCMA"]
    if default_optims is not None:
        optims = default_optims
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def sequential_fastgames(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of policies for games, i.e. direct policy search.
    Budget 12800, 25600, 51200, 102400.
    Games: War, Batawaf, Flip, GuessWho,  BigGuessWho."""
    funcs = [game.Game(name) for name in ["war", "batawaf", "flip", "guesswho", "bigguesswho"]]
    seedg = create_seed_generator(seed)
    optims = ["CMA", "DiagonalCMA", "NGOpt8",
              "OptimisticNoisyOnePlusOne", "OptimisticDiscreteOnePlusOne"]
    optims += get_optimizers("splitters", "progressive", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims
    for budget in [12800, 25600, 51200, 102400]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def powersystems(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Unit commitment problem, i.e. management of dams for hydroelectric planning."""
    funcs: tp.List[ExperimentFunction] = []
    for dams in [3, 5, 9, 13]:
        funcs += [PowerSystem(dams, depth=2, width=3)]
    seedg = create_seed_generator(seed)
    optims = ["NaiveTBPSA", "ScrHammersleySearch", "PSO", "OnePlusOne",
              "CMA", "TwoPointsDE", "QrDE", "LhsDE", "Zero", "StupidRandom", "RandomSearch", "HaltonSearch",
              "RandomScaleRandomSearch", "MiniDE",
              "NGO", "Shiwa", "DiagonalCMA"]
    if default_optims is not None:
        optims = default_optims
    optims += ["OptimisticNoisyOnePlusOne", "OptimisticDiscreteOnePlusOne", "OnePlusOne"]
    optims += get_optimizers("splitters", "progressive", seed=next(seedg))  # type: ignore
    budgets = [3200, 6400, 12800]
    for budget in budgets:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mlda(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed."""
    funcs: tp.List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale) for name, num in [("Ruspini", 5), ("German towns", 10)] for
        rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Perceptron.from_mlda(name) for name in ["quadratic", "sine", "abs", "heaviside"]]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]
    seedg = create_seed_generator(seed)
    optims = ["DiagonalCMA", "MetaModel", "NGOpt8", "CMandAS2", "OnePlusOne", "TwoPointsDE"]
    if default_optims is not None:
        optims = default_optims
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mldakmeans(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed, restricted to the K-means part."""
    funcs: tp.List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale) for name, num in [("Ruspini", 5), ("German towns", 10),
                                                                         ("Ruspini", 50), ("German towns", 100)] for
        rescale in [True, False]
    ]
    seedg = create_seed_generator(seed)

    optims = ["OptimisticNoisyOnePlusOne", "OptimisticDiscreteOnePlusOne", "CMA", "TBPSA", "NaiveTBPSA", "SPSA"]
    optims += get_optimizers("progressive", seed=next(seedg))  # type: ignore
    if default_optims is not None:
        optims = default_optims
    for budget in [1000, 10000]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def arcoating(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """AR coating. Problems about optical properties of nanolayers."""
    seedg = create_seed_generator(seed)
    optims = ["NGOpt8", "PortfolioDiscreteOnePlusOne", "CMA"]
    if default_optims is not None:
        optims = default_optims
    # for budget in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
    for budget in [100 * 5 ** k for k in range(5)]:  # from 100 to 62500
        for num_workers in [1, 10, 100]:
            for algo in optims:
                for func in [ARCoating(10, 400), ARCoating(35, 700), ARCoating(70, 1000)]:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def image_similarity(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    if default_optims is not None:
        optims = default_optims
    func = imagesxp.Image()
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


imagelosses = [imagesxp.imagelosses.SumAbsoluteDifferences,
                imagesxp.imagelosses.LpipsAlex,
                imagesxp.imagelosses.LpipsVgg,
                imagesxp.imagelosses.SumSquareDifferences,
                imagesxp.imagelosses.HistogramDifference]


@registry.register
def image_multi_similarity(seed: tp.Optional[int] = None, cross_valid: bool=False) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    if default_optims is not None:
        optims = default_optims
    funcs = [imagesxp.Image(loss=loss) for loss in [
                imagesxp.imagelosses.SumAbsoluteDifferences,
                imagesxp.imagelosses.LpipsAlex,
                imagesxp.imagelosses.LpipsVgg,
                imagesxp.imagelosses.SumSquareDifferences,
                imagesxp.imagelosses.HistogramDifference]]
    base_values: tp.List[tp.Any] = [func(func.parametrization.sample().value) for func in funcs]
    mofuncs = fbase.multi_experiments(funcs, upper_bounds=base_values) if cross_valid else [
            fbase.MultiExperiment(funcs, upper_bounds=base_values)]
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def image_with_similarity_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_multi_similarity with cross-validation."""
    return image_with_similarity(seed, cross_valid=True)



@registry.register
def image_quality_proxy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo")
    if default_optims is not None:
        optims = default_optims

    
    # We optimize func_blur or func_brisque and check performance on func_iqa.
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur)
    func_brisque = imagesxp.Image(loss=imagesxp.imagelosses.NegBrisque)
    
    # TODO: add the proxy info in the parametrization.
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in [func_blur.copy(), func_brisque.copy()]:
                    func.evaluation_function = func_iqa.evaluation_function
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def image_quality(seed: tp.Optional[int] = None, cross_val: bool=False) -> tp.Iterator[Experiment]:
    """Optimizing images for quality:
    TODO
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo")
    if default_optims is not None:
        optims = default_optims
    
    # We optimize func_blur or func_brisque and check performance on func_iqa.
    funcs = [
            imagesxp.Image(loss=imagesxp.imagelosses.Koncept512),
            imagesxp.Image(loss=imagesxp.imagelosses.Blur),
            imagesxp.Image(loss=imagesxp.imagelosses.NegBrisque),
            ]
    upper_bounds = [func(func.parametrization.sample().value) for func in funcs]
    # TODO: add the proxy info in the parametrization.
    for budget in [100 * 5 ** k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                mofuncs = fbase.multi_experiments(funcs, upper_bounds=upper_bounds, no_crossval=[1, 2]) if cross_val else [fbase.MultiExperiment(funcs, upper_bounds=upper_bounds)]
                for func in mofuncs:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def image_quality_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, cross_val=True)


@registry.register
def image_similarity_and_quality(seed: tp.Optional[int] = None, cross_val: bool=False) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo")
    if default_optims is not None:
        optims = default_optims

    
    # 3 losses functions including 2 iqas.
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur)
    base_blur_value = func_blur(func_blur.parametrization.sample().value)
    for func in [imagesxp.Image(loss=loss) for loss in imagelosses]:
    
        # Creating a reference value.
        base_value = func(func.parametrization.sample().value)
        mofunc = fbase.multi_experiments([func, func_blur, func_iqa], upper_bounds=[base_value, base_blur_value, 100.], no_crossval=[1, 2]) if cross_val else [
                fbase.MultiExperiment([func, func_blur, func_iqa], upper_bounds=[base_value, base_blur_value, 100.])]
        for budget in [100 * 5 ** k for k in range(3)]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def image_similarity_and_quality_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_multi_similarity with cross-validation."""
    return image_similarity_and_quality(seed, cross_valid=True)


# TODO: GAN counterparts of the above ?
@registry.register
def images_using_gan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimizing an image using koncept512 and a GAN"""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo")
    func = imagesxp.ImageFromPGAN()
    num_workers = 1
    for budget in [100 * 5 ** k for k in range(3)]:
        for algo in optims:
            xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
            if not xp.is_incoherent:
                yield xp


@registry.register
def double_o_seven(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of policies for the 007 game.
    Sequential or 10-parallel or 100-parallel. Various numbers of averagings: 1, 10 or 100."""
    # pylint: disable=too-many-locals
    seedg = create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {'mono': rl.agents.Perceptron, 'multi': rl.agents.DenseNet}
    agents = {a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False) for a, m in modules.items()}
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    for num_repetitions in [1, 10, 100]:
        for archi in ["mono", "multi"]:
            dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
            for optim in ["PSO", dde, "MetaTuneRecentering", "DiagonalCMA"]:
                for env_budget in [5000, 10000, 20000, 40000]:
                    for num_workers in [1, 10, 100]:
                        # careful, not threadsafe
                        runner = rl.EnvironmentRunner(env.copy(), num_repetitions=num_repetitions, max_step=50)
                        func = rl.agents.TorchAgentFunction(agents[archi], runner,
                                                            reward_postprocessing=lambda x: 1 - x)
                        opt_budget = env_budget // num_repetitions
                        yield Experiment(func, optim, budget=opt_budget, num_workers=num_workers,  # type: ignore
                                         seed=next(seedg))  # type: ignore


@registry.register
def multiobjective_example(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of 2 and 3 objective functions in Sphere, Ellipsoid, Cigar, Hm.
    Dimension 6 and 7.
    Budget 100 to 3200
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = ["NaiveTBPSA", "PSO", "DE", "LhsDE", "RandomSearch", "NGO", "Shiwa", "DiagonalCMA",
                               "CMA", "OnePlusOne", "TwoPointsDE", "NGOpt8"]
    optims += [ng.families.DifferentialEvolution(multiobjective_adaptation=False).set_name("DE-noadapt"),
               ng.families.DifferentialEvolution(crossover="twopoints", multiobjective_adaptation=False).set_name(
                   "TwoPointsDE-noadapt")]
    mofuncs: tp.List[fbase.MultiExperiment] = []
    for name1, name2 in itertools.product(["sphere"], ["sphere", "hm"]):
        mofuncs.append(fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=7),
                                              ArtificialFunction(name2, block_dimension=7)],
                                             upper_bounds=[100, 100]))
        mofuncs.append(fbase.MultiExperiment([ArtificialFunction(name1, block_dimension=6),
                                              ArtificialFunction("sphere", block_dimension=6),
                                              ArtificialFunction(name2, block_dimension=6)],
                                             upper_bounds=[100, 100, 100.]))
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1, 100]:
                    yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def pbt(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optimizers = ["CMA", "TwoPointsDE", "Shiwa", "OnePlusOne", "DE", "PSO", "NaiveTBPSA",
                  "RecombiningOptimisticNoisyDiscreteOnePlusOne", "PortfolioNoisyDiscreteOnePlusOne"]  # type: ignore
    for func in PBT.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def far_optimum_es(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    popsizes = [5, 40]
    es = [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=False, popsize=pop)
          for recomb in [0, 1] for pop in popsizes]
    es += [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=only, popsize=pop,
                                         offsprings=10 if pop == 5 else 60)
           for only in [True, False] for recomb in [0, 1] for pop in popsizes]
    optimizers = ["CMA", "TwoPointsDE", "NGOpt8"] + es  # type: ignore
    for func in FarOptimumFunction.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def photonics(seed: tp.Optional[int] = None, as_tuple: bool = False) -> tp.Iterator[Experiment]:
    """Too small for being interesting: Bragg mirror + Chirped + Morpho butterfly."""
    seedg = create_seed_generator(seed)
    popsizes = [20, 40, 80]
    es = [ng.families.EvolutionStrategy(recombination_ratio=recomb, only_offsprings=only, popsize=pop,
                                        offsprings=pop * 5)
          for only in [True, False] for recomb in [0.1, .5] for pop in popsizes]
    optims = ["TwoPointsDE", "DE", "PSO", "NGOpt8"] + es  # type: ignore

    optims += get_optimizers("splitters")  # type: ignore
    if default_optims is not None:
        optims = default_optims
    for method in ["clipping", "tanh"]:  # , "arctan"]:
        for name in ["bragg", "chirped", "morpho"]:
            func = Photonics(name, 60 if name == "morpho" else 80, bounding_method=method, as_tuple=as_tuple)
            for budget in [1e3, 1e4, 1e5, 1e6]:
                for algo in optims:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def photonics2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=True)


@registry.register
def bragg_structure(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Too small for being interesting: Bragg mirror."""
    seedg = create_seed_generator(seed)
    recombinable: tp.List[tp.Union[str, ConfiguredOptimizer]] = [
        ng.families.EvolutionStrategy(recombination_ratio=0.1, popsize=40).set_name("Pairwise-ES"),
        ng.families.DifferentialEvolution(crossover="parametrization").set_name("Param-DE")
    ]
    optims = ["TwoPointsDE", "DE", "CMA", "NaiveTBPSA", "DiagonalCMA", "NGOpt8", "OnePlusOne"]
    optims += get_optimizers("splitters")  # type: ignore
    func = Photonics("bragg", 80, bounding_method="clipping")
    func.parametrization.set_name("layer")
    #
    func_nostruct = Photonics("bragg", 80, bounding_method="clipping")
    func_nostruct.parametrization.set_name("2pt").set_recombination(ng.p.mutation.RavelCrossover())  # type: ignore
    #
    func_mix = Photonics("bragg", 80, bounding_method="clipping")
    param = func_mix.parametrization
    param.set_name("mix")
    param.set_recombination(  # type: ignore
        ng.p.Choice([ng.p.mutation.Crossover(axis=1), ng.p.mutation.RavelCrossover()]))  # type: ignore
    muts = ["gaussian", "cauchy", ng.p.mutation.Jumping(axis=1, size=5), ng.p.mutation.Translation(axis=1)]
    muts += [ng.p.mutation.LocalGaussian(axes=1, size=10)]
    param.set_mutation(custom=ng.p.Choice(muts))  # type: ignore
    for budget in [1e3, 1e4, 1e5, 1e6]:
        xpseed = next(seedg)
        for algo in default_optims if default_optims is not None else optims:
            yield Experiment(func, algo, int(budget), num_workers=1, seed=xpseed)
        for f in [func, func_nostruct, func_mix]:
            for algo in recombinable:  # type: ignore
                yield Experiment(f, algo, int(budget), num_workers=1, seed=xpseed)


@registry.register
def adversarial_attack(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Pretrained ResNes50 under black-box attacked.
    Square attacks:
    100 queries ==> 0.1743119266055046
    200 queries ==> 0.09043250327653997
    300 queries ==> 0.05111402359108781
    400 queries ==> 0.04325032765399738
    1700 queries ==> 0.001310615989515072
    """
    seedg = create_seed_generator(seed)
    optims = ["DiagonalCMA", "MetaModel", "CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    folder = os.environ.get("NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER", None)
    if folder is None:
        warnings.warn("Using random images, set variable NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER to specify a folder")
    for func in imagesxp.ImageAdversarial.make_folder_functions(folder=folder):
        for budget in [100, 200, 300, 400, 1700]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def pbo_suite(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # Discrete, unordered.
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    seedg = create_seed_generator(seed)
    for dim in [16, 64, 100]:
        for fid in range(1, 24):
            for iid in range(1, 5):
                try:
                    func = iohprofiler.PBOFunction(fid, iid, dim)
                except ModuleNotFoundError as e:
                    raise fbase.UnsupportedExperiment("IOHexperimenter needs to be installed") from e
                for optim in ["DiscreteOnePlusOne", "Shiwa", "CMA", "PSO", "TwoPointsDE", "DE", "OnePlusOne", "AdaptiveDiscreteOnePlusOne",
                              "CMandAS2", "PortfolioDiscreteOnePlusOne", "DoubleFastGADiscreteOnePlusOne", "MultiDiscrete", "cGA", dde]:
                    for nw in [1, 10]:
                        for budget in [100, 1000, 10000]:
                            yield Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))  # type: ignore
