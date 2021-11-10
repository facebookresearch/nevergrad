# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
from nevergrad.functions import gym
from nevergrad.functions import ExperimentFunction
from .xpbase import registry
from .xpbase import create_seed_generator
from .xpbase import Experiment
from .optgroups import get_optimizers

# pylint: disable=too-many-nested-blocks,stop-iteration-return


@registry.register
def ng_full_gym(
    seed: tp.Optional[int] = None,
    randomized: bool = True,
    multi: bool = False,
    big: bool = False,
    memory: bool = False,
    ng_gym: bool = False,  # pylint: disable=redefined-outer-name
    conformant: bool = False,
) -> tp.Iterator[Experiment]:
    """Gym simulator. Maximize reward.  Many distinct problems.

    Parameters:
        seed: int
           random seed.
        randomized: bool
           whether we keep the problem's stochasticity
        multi: bool
           do we have one neural net per time step
        big: bool
           do we consider big budgets
        memory: bool
           do we use recurrent nets
        ng_gym: bool
           do we restrict to ng-gym
        conformant: bool
           do we restrict to conformant planning, i.e. deterministic controls.
    """
    env_names = gym.GymMulti.get_env_names()
    if ng_gym:
        env_names = gym.GymMulti.ng_gym
    seedg = create_seed_generator(seed)
    optims = ["DiagonalCMA", "OnePlusOne", "PSO", "DiscreteOnePlusOne", "DE", "CMandAS2"]
    if multi:
        controls = ["multi_neural"]
    else:
        controls = (
            [
                "neural",
                "structured_neural",
                # "memory_neural",
                "stackingmemory_neural",
                "deep_neural",
                "semideep_neural",
                # "noisy_neural",
                # "noisy_scrambled_neural",
                # "scrambled_neural",
                # "linear",
            ]
            if not big
            else ["neural"]
        )
    if memory:
        controls = ["stackingmemory_neural", "deep_stackingmemory_neural", "semideep_stackingmemory_neural"]
        controls += ["memory_neural", "deep_memory_neural", "semideep_memory_neural"]
        controls += [
            "extrapolatestackingmemory_neural",
            "deep_extrapolatestackingmemory_neural",
            "semideep_extrapolatestackingmemory_neural",
        ]
        assert not multi
    if conformant:
        controls = ["stochastic_conformant"]
    budgets = [50, 200, 800, 3200, 6400, 100, 25, 400, 1600]
    for control in controls:
        neural_factors: tp.Any = (
            [None]
            if (conformant or control == "linear")
            else ([1] if "memory" in control else [3 if big else 1])
        )
        for neural_factor in neural_factors:
            for name in env_names:
                try:
                    func = gym.GymMulti(
                        name, control=control, neural_factor=neural_factor, randomized=randomized
                    )
                except MemoryError:
                    continue
                for budget in budgets:
                    for algo in optims:
                        xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def multi_ng_full_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with one neural net per time step.

    Each neural net is used for many problems, but only for one of the time steps."""
    return ng_full_gym(seed, multi=True)


@registry.register
def conformant_ng_full_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with fixed, predetermined actions for each time step.

    This is conformant: we optimize directly the actions for a given context.
    This does not prevent stochasticity, but actions do not depend on observationos.
    """
    return ng_full_gym(seed, conformant=True)


@registry.register
def ng_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with a specific, reduced list of problems."""
    return ng_full_gym(seed, ng_gym=True)


@registry.register
def ng_stacking_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_gym with a recurrent network."""
    return ng_full_gym(seed, ng_gym=True, memory=True)


@registry.register
def big_gym_multi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with bigger nets."""
    return ng_full_gym(seed, big=True)


@registry.register
def deterministic_gym_multi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with fixed seeds (so that the problem becomes deterministic)."""
    return ng_full_gym(seed, randomized=False)


# Not registered because not validated.
def gym_multifid_anm(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Gym simulator for Active Network Management."""

    func = gym.GymMulti("multifidLANM")
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", "progressive", "splitters", "baselines", seed=next(seedg))
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


def gym_problem(
    seed: tp.Optional[int] = None,
    specific_problem: str = "LANM",
    conformant: bool = False,
    compiler_gym_pb_index: tp.Optional[int] = None,
    limited_compiler_gym: tp.Optional[bool] = None,
    big_noise: bool = False,
    multi_scale: bool = False,
    greedy_bias: bool = False,
) -> tp.Iterator[Experiment]:
    """Gym simulator for Active Network Management (default) or other pb.

    seed: int
        random seed for determinizing the problem
    specific_problem: string
        name of the problem we are working on
    conformant: bool
        do we focus on conformant planning
    compiler_gym_pb_index: integer
        index of Uris problem we work on.
    limited_compiler_gym: boolean
        for compiler-gyn, whether we use a restricted action space
    big_noise: bool
        do we switch to specific optimizers, dedicated to noise
    multi_scale: boolean
        do we check multiple scales
    greedy_bias: boolean
        do we use greedy reward estimates for biasing the decisions.
    """
    if "directcompilergym" in specific_problem:
        assert compiler_gym_pb_index is not None
        assert limited_compiler_gym is not None
        assert compiler_gym_pb_index >= 0
        assert greedy_bias is False
        funcs: tp.List[ExperimentFunction] = [
            gym.CompilerGym(
                compiler_gym_pb_index=compiler_gym_pb_index, limited_compiler_gym=limited_compiler_gym
            )
        ]
    else:
        if conformant:
            funcs = [
                gym.GymMulti(
                    specific_problem,
                    control="conformant",
                    limited_compiler_gym=limited_compiler_gym,
                    compiler_gym_pb_index=compiler_gym_pb_index,
                    neural_factor=None,
                )
            ]
        else:
            funcs = [
                gym.GymMulti(
                    specific_problem,
                    control=control,
                    neural_factor=1 if control != "linear" else None,
                    limited_compiler_gym=limited_compiler_gym,
                    optimization_scale=scale,
                    greedy_bias=greedy_bias,
                )
                for scale in ([-6, -4, -2, 0] if multi_scale else [0])
                for control in (
                    ["deep_neural", "semideep_neural", "neural", "linear"] if not greedy_bias else ["neural"]
                )
            ]
    seedg = create_seed_generator(seed)
    optims = [
        "TwoPointsDE",
        "GeneticDE",
        "PSO",
        "DiagonalCMA",
    ]
    if "stochastic" in specific_problem:
        optims = ["DiagonalCMA", "TBPSA"] if big_noise else ["DiagonalCMA"]
    if specific_problem == "EnergySavingsGym" and conformant:  # Do this for all conformant discrete ?
        optims = [
            "DiscreteOnePlusOne",
            "PortfolioDiscreteOnePlusOne",
            "DiscreteLenglerOnePlusOne",
            "AdaptiveDiscreteOnePlusOne",
            "AnisotropicAdaptiveDiscreteOnePlusOne",
            "DiscreteBSOOnePlusOne",
            "DiscreteDoerrOnePlusOne",
            "OptimisticDiscreteOnePlusOne",
            "NoisyDiscreteOnePlusOne",
            "DoubleFastGADiscreteOnePlusOne",
            "SparseDoubleFastGADiscreteOnePlusOne",
            "RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne",
            "RecombiningPortfolioDiscreteOnePlusOne",
            "MultiDiscrete",
            "NGOpt",
        ]
    for func in funcs:
        for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
            for num_workers in [1]:
                if num_workers < budget:
                    for algo in optims:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def limited_stochastic_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. Stochastic problem: we are optimizing a net for driving compilation."""
    return gym_problem(seed, specific_problem="stochasticcompilergym", limited_compiler_gym=True)


@registry.register
def multiscale_limited_stochastic_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. Stochastic problem: we are optimizing a net for driving compilation."""
    return gym_problem(
        seed, specific_problem="stochasticcompilergym", limited_compiler_gym=True, multi_scale=True
    )


@registry.register
def unlimited_hardcore_stochastic_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. Stochastic problem: we are optimizing a net for driving compilation."""
    return gym_problem(
        seed, specific_problem="stochasticcompilergym", limited_compiler_gym=False, big_noise=True
    )


@registry.register
def conformant_planning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    specific_problem = "EnergySavingsGym"
    # You might modify this problem by specifying an environment variable.
    if os.environ.get("TARGET_GYM_ENV") is not None:
        specific_problem = os.environ.get("TARGET_GYM_ENV")  # type: ignore
    return gym_problem(
        seed,
        specific_problem=specific_problem,
        conformant=True,
        big_noise=False,
    )


@registry.register
def neuro_planning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    specific_problem = "EnergySavingsGym"
    # You might modify this problem by specifying an environment variable.
    if os.environ.get("TARGET_GYM_ENV") is not None:
        specific_problem = os.environ.get("TARGET_GYM_ENV")  # type: ignore
    return gym_problem(
        seed,
        specific_problem=specific_problem,
        conformant=False,
        big_noise=False,
    )


@registry.register
def limited_hardcore_stochastic_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. Stochastic problem: we are optimizing a net for driving compilation."""
    return gym_problem(
        seed, specific_problem="stochasticcompilergym", limited_compiler_gym=True, big_noise=True
    )


@registry.register
def greedy_limited_stochastic_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. Stochastic problem: we are optimizing a net for driving compilation."""
    return gym_problem(
        seed, specific_problem="stochasticcompilergym", limited_compiler_gym=True, greedy_bias=True
    )


@registry.register
def unlimited_stochastic_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. Stochastic problem: we are optimizing a net for driving compilation."""
    return gym_problem(seed, specific_problem="stochasticcompilergym", limited_compiler_gym=False)


@registry.register
def unlimited_direct_problems23_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. All 23 problems."""
    for compiler_gym_pb_index in range(23):
        pb = gym_problem(
            seed,
            specific_problem="directcompilergym" + str(compiler_gym_pb_index),
            compiler_gym_pb_index=compiler_gym_pb_index,
            limited_compiler_gym=False,
        )
        for xp in pb:
            yield xp


@registry.register
def limited_direct_problems23_compiler_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Working on CompilerGym. All 23 problems."""
    for compiler_gym_pb_index in range(23):
        pb = gym_problem(
            seed,
            specific_problem="directcompilergym" + str(compiler_gym_pb_index),
            compiler_gym_pb_index=compiler_gym_pb_index,
            limited_compiler_gym=True,
        )
        for xp in pb:
            yield xp
