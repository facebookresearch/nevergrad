# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import typing as tp
from nevergrad.functions import gym as nevergrad_gym
from .xpbase import registry
from .xpbase import create_seed_generator
from .xpbase import Experiment
from .optgroups import get_optimizers

# pylint: disable=too-many-nested-blocks,stop-iteration-return


# A few GYM modifiers based on environment variables.
def gym_problem_modifier(specific_problem):
    specific_problem = os.environ.get("TARGET_GYM_ENV", specific_problem)
    print("problem=", specific_problem)
    return specific_problem


def gym_optimizer_modifier(optims):
    print(optims)
    if os.environ.get("GYM_OPTIMIZER") is not None:
        optimizer_string = os.environ.get("GYM_OPTIMIZER")
        print(f"Considering optimizers with {optimizer_string} in their name.")
        optims = [o for o in optims if optimizer_string in str(o)]  # type: ignore
        if len(optims) == 0:
            optims = [optimizer_string]  # type: ignore
    print("optims=", optims)
    return optims


def gym_budget_modifier(budgets):
    if os.environ.get("MAX_GYM_BUDGET") is not None:
        budget_string = os.environ.get("MAX_GYM_BUDGET")
        budgets = [b for b in budgets if b < int(budget_string)]  # type: ignore
    print("budgets=", budgets)
    return budgets


@registry.register
def ng_full_gym(
    seed: tp.Optional[int] = None,
    randomized: bool = True,
    multi: bool = False,
    big: bool = False,
    memory: bool = False,
    ng_gym: bool = False,  # pylint: disable=redefined-outer-name
    conformant: bool = False,
    gp: bool = False,
    sparse: bool = False,
    multi_scale: bool = False,
    small: bool = False,
    tiny: bool = False,
    structured: bool = False,
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
    env_names = nevergrad_gym.GymMulti.get_env_names()
    assert int(ng_gym) + int(gp) <= 1, "At most one specific list of environments."
    if ng_gym:
        env_names = nevergrad_gym.GymMulti.ng_gym
    if gp:
        import pybullet_envs  # pylint: disable=unused-import

        env_names = [
            "CartPole-v1",
            "Acrobot-v1",
            "MountainCarContinuous-v0",
            "Pendulum-v1",
            # "InvertedPendulumSwingupBulletEnv-v0",
            "BipedalWalker-v3",
            "BipedalWalkerHardcore-v3",
            "HopperBulletEnv-v0",
            # "InvertedDoublePendulumBulletEnv-v0",
            "LunarLanderContinuous-v2",
        ]

    seedg = create_seed_generator(seed)
    optims = [
        "DiagonalCMA",
        "GeneticDE",
        "NoisyRL1",
        "NoisyRL2",
        "NoisyRL3",
        "MixDeterministicRL",
        "SpecialRL",
        "PSO",
        "NGOpt",
        "NgIoh21",
    ]
    optims = [
        np.random.choice(
            [
                "NgIoh21",
                "NgIoh21",
                "NgIoh21",
                "Carola4",
                "Carola5",
                "Carola6",
                "Carola8",
                "Carola9",
                "Carola14",
            ]
        )
    ]  # DiagonalCMA", "CMA","NgIoh21", "NgIoh21"])] #"PSO", "SQOPSO", "NGOpt", "NGOptRW", "NoisyRL1", "NoisyRL3", "NoisyRL2", "SpecialRL", "NGDSRW", "NgIoh21"])]
    optims = ["NGOpt", "NgDS", "NgDS2", "NGDSRW"]
    optims = ["NgIoh4", "NgDS2", "NgDS3"]
    optims = ["NgDS3", "NGDSRW"]
    optims = ["CSEC11"]
    optims = [np.random.choice(["NGOpt", "SQOPSO", "CMA", "CSEC11", "CMA", "VLPCMA", "DE", "QODE"])]
    optims = ["SQOPSODCMA"]
    optims = [
        np.random.choice(
            ["CMA", "PSO", "CSEC11", "NgIoh4", "RandomSearch", "DiagonalCMA", "CMA", "VLPCMA", "Cobyla"]
        )
    ]
    # optims = [np.random.choice(["TBPSA", "NoisyRL2", "NoisyRL3", "SpecialRL", "MixDeterministicRL"])]
    optims = [np.random.choice(["QODE", "QNDE"])]
    optims = ["NgIoh4"]
    optims = [np.random.choice(["CSEC11", "SQOPSODCMA", "NgIoh4", "NGOpt"])]
    optims = ["CSEC11"]
    if structured:
        optims = get_optimizers("split", seed=next(seedg))  # type: ignore
        # optims = [np.random.choice(optims)]
    optims = [
        np.random.choice(
            [
                "BigLognormalDiscreteOnePlusOne",
                "XLognormalDiscreteOnePlusOne",
                "SmallLognormalDiscreteOnePlusOne",
                "HugeLognormalDiscreteOnePlusOne",
            ]
        )
    ]
    optims = ["OLNDiscreteOnePlusOne"]
    optims = ["NgIohLn"]
    optims = [np.random.choice(["NgIohMLn", "NgIohLn", "NgIoh"])]
    if multi:
        controls = ["multi_neural"]
    else:
        controls = (
            [
                "noisy_semideep_neural",
                "noisy_scrambled_semideep_neural",  # Scrambling: why not perturbating the order of variables ?
                "noisy_deep_neural",
                "noisy_scrambled_deep_neural",
                "neural",
                # "structured_neural",
                # "memory_neural",
                "stackingmemory_neural",
                "deep_neural",
                "semideep_neural",
                "noisy_neural",
                "noisy_scrambled_neural",
                # "scrambled_neural",
                # "linear",
                "resid_neural",
                "resid_semideep_neural",
                "resid_deep_neural",
            ]
            if not big
            else ["resid_neural"]
        )
    if structured:
        controls = ["neural", "structured_neural"]
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
    optimization_scales: tp.List[int] = [0]
    if multi_scale:
        optimization_scales = [-6, -4, -2, 0]
    budgets = [50, 200, 100, 25, 400]  # Let's go with low budget.
    budgets = gym_budget_modifier(budgets)
    for control in controls:
        neural_factors: tp.Any = (
            [None]
            if (conformant or control == "linear")
            else ([1] if "memory" in control else ([3] if big else ([1] if (tiny or small) else [1, 2, 3])))
        )
        for neural_factor in neural_factors:
            for name in env_names:
                sparse_limits: tp.List[tp.Optional[int]] = [None]
                if sparse:
                    sparse_limits += [10, 100, 1000]
                for sparse_limit in sparse_limits:
                    for optimization_scale in optimization_scales:
                        try:
                            func = nevergrad_gym.GymMulti(
                                name,
                                control=control,
                                neural_factor=neural_factor,
                                randomized=randomized,
                                optimization_scale=optimization_scale,
                                sparse_limit=sparse_limit,
                            )
                            if not randomized:
                                func.parametrization.function.deterministic = True
                                func.parametrization.enforce_determinism = True
                        except MemoryError:
                            continue
                        for budget in budgets:
                            for algo in optims:
                                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                                xp.function.parametrization.real_world = True
                                xp.function.parametrization.neural = True
                                if xp.function.parametrization.dimension > 40 and small:
                                    continue
                                if xp.function.parametrization.dimension > 20 and tiny:
                                    continue
                                #    continue
                                if not xp.is_incoherent:
                                    yield xp


@registry.register
def multi_ng_full_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with one neural net per time step.

    Each neural net is used for many problems, but only for one of the time steps."""
    return ng_full_gym(seed, multi=True)


@registry.register
def multi_structured_ng_full_gym(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with one neural net per time step.

    Each neural net is used for many problems, but only for one of the time steps."""
    return ng_full_gym(seed, multi=True, structured=True)


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
def gp(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """GP benchmark.

    Counterpart of ng_full_gym with a specific, reduced list of problems for matching
    a genetic programming benchmark."""
    return ng_full_gym(seed, gp=True, multi_scale=True)


@registry.register
def conformant_gp(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """GP benchmark.

    Counterpart of ng_full_gym with a specific, reduced list of problems for matching
    a genetic programming benchmark."""
    return ng_full_gym(seed, conformant=True, gp=True)


@registry.register
def sparse_gp(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """GP benchmark.

    Counterpart of ng_full_gym with a specific, reduced list of problems for matching
    a genetic programming benchmark."""
    return ng_full_gym(seed, gp=True, sparse=True, multi_scale=True)


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


@registry.register
def tiny_deterministic_gym_multi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with fixed seeds (so that the problem becomes deterministic)."""
    return ng_full_gym(seed, randomized=False, tiny=True)


@registry.register
def small_deterministic_gym_multi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of ng_full_gym with fixed seeds (so that the problem becomes deterministic)."""
    return ng_full_gym(seed, randomized=False, small=True)


# Not registered because not validated.
def gym_multifid_anm(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Gym simulator for Active Network Management."""

    func = nevergrad_gym.GymMulti("multifidLANM")
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
    big_noise: bool
        do we switch to specific optimizers, dedicated to noise
    multi_scale: boolean
        do we check multiple scales
    greedy_bias: boolean
        do we use greedy reward estimates for biasing the decisions.
    """
    if conformant:
        funcs = [
            nevergrad_gym.GymMulti(
                specific_problem,
                control="conformant",
                neural_factor=None,
            )
        ]
    else:
        funcs = [
            nevergrad_gym.GymMulti(
                specific_problem,
                control=control,
                neural_factor=1 if control != "linear" else None,
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
        "DoubleFastGADiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "PortfolioDiscreteOnePlusOne",
        "MixDeterministicRL",
        "NoisyRL2",
        "NoisyRL3",
        "SpecialRL",
        "NGOpt39",
        "CMA",
        "DE",
    ]
    if "stochastic" in specific_problem:
        optims = ["DiagonalCMA", "TBPSA"] if big_noise else ["DiagonalCMA"]
    if specific_problem == "EnergySavingsGym-v0" and conformant:  # Do this for all conformant discrete ?
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

    optims = gym_optimizer_modifier(optims)
    budgets = [25, 50, 100, 200]
    budgets = gym_budget_modifier(budgets)
    for func in funcs:
        for budget in budgets:
            for num_workers in [1]:
                if num_workers < budget:
                    for algo in optims:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        xp.function.parametrization.neural = not conformant

                        if not xp.is_incoherent:
                            yield xp


@registry.register
def conformant_planning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # You might modify this problem by specifying an environment variable.
    specific_problem = "EnergySavingsGym-v0"

    return gym_problem(
        seed,
        specific_problem=gym_problem_modifier(specific_problem),
        conformant=True,
        big_noise=False,
    )


@registry.register
def neuro_planning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # You might modify this problem by specifying an environment variable.
    specific_problem = "EnergySavingsGym-v0"
    return gym_problem(
        seed,
        specific_problem=gym_problem_modifier(specific_problem),
        conformant=False,
        big_noise=False,
    )
