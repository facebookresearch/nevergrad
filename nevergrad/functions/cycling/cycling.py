# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Creator: Ryan Kroon, Email: rkroon19@gmail.com
# Accompanying paper: Extending Nevergrad, an Optimisation Platform (in directory)

# Evolving Pacing Strategies for Team Pursuit Track Cycling
# Markus Wagner, Jareth Day, Diora Jordan, Trent Kroeger, Frank Neumann
# Advances in Metaheuristics, Vol. 53, Springer, 2013.
# https://link.springer.com/chapter/10.1007/978-1-4614-6322-1_4
# Java code: https://cs.adelaide.edu.au/~markus/pub/TeamPursuit.zip

import numpy as np
import typing as tp
from .. import base
from nevergrad.parametrization import parameter as p
from .womensteampursuit import womensteampursuit
from .mensteampursuit import mensteampursuit


class Cycling(base.ExperimentFunction):
    """
    Team Pursuit Track Cycling Simulator.

    Parameters
    ----------
    strategy: int
        Refers to Transition strategy or Pacing strategy (or both) of the cyclists; this depends on the strategy length.
        Strategy length can only be 30, 31, 61, 22, 23, 45.
        30: mens transition strategy.
        31: mens pacing strategy.
        61: mens transition and pacing strategy combined.
        22: womens transition strategy.
        23: womens pacing strategy.
        45: womens transition and pacing strategy combined.
    """

    def __init__(self, strategy_index: int = 30) -> None:

        # Preliminary stuff.
        women = strategy_index in [22, 23, 45]
        param_transition = p.TransitionChoice([False, True], repetitions=22 if women else 30)
        init = (400 if women else 550) * np.ones(23 if women else 31)
        gender = "Women" if women else "Men"
        param_pacing = p.Array(init=init, lower=200, upper=1200)
        target_function = team_pursuit_simulation

        # optimising transition strategy
        if strategy_index in (22, 30):
            parameter: tp.Any = param_transition
            parameter.set_name(f"{gender} Transition strategy")
            parameter.set_name("")

        # optimising pacing strategy
        elif strategy_index in (23, 31):
            init = 550 * np.ones(strategy_index)
            parameter = param_pacing
            parameter.set_name(f"{gender} Pacing strategy")
            parameter.set_name("")

        # optimising pacing and transition strategies
        elif strategy_index in (45, 61):
            init = 0.5 * np.ones(strategy_index)  # type: ignore
            parameter = p.Instrumentation(transition=param_transition, pacing=param_pacing)
            parameter.set_name(f"{gender} Pacing and Transition")
            # For some reason the name above does not work...
            # It generates a very long name like
            # "(Wom|M)ens Pacing and Transition:[0.5,...
            parameter.set_name("")

        # error raised if invalid strategy length given
        else:
            raise ValueError("Strategy length must be any of 22, 23, 30, 31, 45, 61")
        super().__init__(target_function, parameter)
        # assert len(self.parametrization.sample().value) == strategy_index, f"{len(self.parametrization.sample().value)} != {strategy_index} (init={init} with len={len(init)})."


def team_pursuit_simulation(x) -> float:

    if len(x) == 2:  # Let us concatenate the instrumentation.
        pacing = x[1]["pacing"]
        transition = x[1]["transition"]
    elif len(x) in (30, 22):
        transition = x
        pacing = [550 if len(x) == 30 else 400] * (len(x) + 1)
    elif len(x) in (31, 23):
        pacing = x
        transition = [True, False] * ((len(x) - 1) // 2)
    else:
        raise ValueError(f"len(x) == {len(x)}")

        # What is this ?
        # for i in range(0, len(pacing_strategy)):
        #    pacing_strategy[i] = 100 * pacing_strategy[i] + 200

    # Create a mensteampursuit oor womensteampursuit object.
    team_pursuit: tp.Any = womensteampursuit() if len(pacing) == 23 else mensteampursuit()
    assert len(pacing) in (23, 31)

    # Simulate event with the default strategies
    result = team_pursuit.simulate(transition, pacing)

    if result.get_finish_time() > 10000:  # in case of inf
        return 10000
    else:
        return float(result.get_finish_time())
