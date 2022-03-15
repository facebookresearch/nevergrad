# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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

from .womensteampursuit import womensteampursuit
from .mensteampursuit import mensteampursuit
import random
import numpy as np
import nevergrad as ng
from nevergrad.parametrization import parameter as p
from .. import base


class cycling(base.ExperimentFunction):
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

    def __init__(self, strategy: int = 30) -> None:

        # optimising transition strategy for men's team
        if strategy == 30:
            strategy = p.Choice([False, True], repetitions=strategy)
            parameter = p.Instrumentation(strategy).set_name("")
            super().__init__(mens_team_pursuit_simulation, parameter)

        # optimising pacing strategy for men's team
        elif strategy == 31:
            init = 550 * np.ones(strategy)
            parameter = p.Array(init=init, lower=200, upper=1200)
            parameter.set_name("Mens Pacing strategy")
            super().__init__(mens_team_pursuit_simulation, parameter)

        # optimising pacing and transition strategies for men's team
        elif strategy == 61:
            init = 0.5 * np.ones(strategy)
            parameter = p.Array(init=init, lower=0, upper=1)
            parameter.set_name("Pacing and Transition")
            super().__init__(mens_team_pursuit_simulation, parameter)

        # optimising transition strategy for women's team
        elif strategy == 22:
            strategy = ng.p.Choice([False, True], repetitions=strategy)
            parameter = ng.p.Instrumentation(strategy).set_name("")
            super().__init__(womens_team_pursuit_simulation, parameter)

        # optimising pacing strategy for women's team
        elif strategy == 23:
            init = 400 * np.ones(strategy)
            parameter = p.Array(init=init, lower=200, upper=1200)
            parameter.set_name("Womens Pacing strategy")
            super().__init__(womens_team_pursuit_simulation, parameter)

        # optimising pacing and transition strategies for women's team
        elif strategy == 45:
            init = 0.5 * np.ones(strategy)
            parameter = p.Array(init=init, lower=0, upper=1)
            parameter.set_name("Pacing and Transition")
            super().__init__(womens_team_pursuit_simulation, parameter)

        # error raised if invalid strategy length given
        else:
            raise ValueError("Strategy length must be any of 22, 23, 30, 31, 45, 61")


def mens_team_pursuit_simulation(x: np.ndarray) -> float:

    if len(x) == 30:
        mens_transition_strategy = x
        mens_pacing_strategy = [550] * 31

    elif len(x) == 31:
        mens_transition_strategy = [True, False] * 15
        mens_pacing_strategy = x

    elif len(x) == 45:
        mens_transition_strategy = x[:30]
        for i in range(0, len(mens_transition_strategy)):
            if mens_transition_strategy[i] < 0.5:
                mens_transition_strategy[i] = False
            elif mens_transition_strategy[i] > 0.5:
                mens_transition_strategy[i] = True
            elif mens_transition_strategy[i] == 0.5:
                mens_transition_strategy[i] = random.choice([True, False])

        mens_pacing_strategy = x[30:]
        for i in range(0, len(mens_pacing_strategy)):
            mens_pacing_strategy[i] = 100 * mens_pacing_strategy[i] + 200

    # Create a mensteampursuit object
    mens_team_pursuit = mensteampursuit()

    # Simulate event with the default strategies
    result = mens_team_pursuit.simulate(mens_transition_strategy, mens_pacing_strategy)

    # print(result.get_finish_time())

    if result.get_finish_time() > 10000:  # in case of inf
        return 10000
    else:
        return float(result.get_finish_time())


def womens_team_pursuit_simulation(x: np.ndarray) -> float:

    if len(x) == 22:
        womens_transition_strategy = x
        womens_pacing_strategy = [400] * 23

    elif len(x) == 23:
        womens_transition_strategy = [True, False] * 11
        womens_pacing_strategy = x

    elif len(x) == 45:
        womens_transition_strategy = x[:22]
        for i in range(0, len(womens_transition_strategy)):
            if womens_transition_strategy[i] < 0.5:
                womens_transition_strategy[i] = False
            elif womens_transition_strategy[i] > 0.5:
                womens_transition_strategy[i] = True
            elif womens_transition_strategy[i] == 0.5:
                womens_transition_strategy[i] = random.choice([True, False])

        womens_pacing_strategy = x[22:]
        for i in range(0, len(womens_pacing_strategy)):
            womens_pacing_strategy[i] = 100 * womens_pacing_strategy[i] + 200

    # Create a womensteampursuit object
    womens_team_pursuit = womensteampursuit()

    # Simulate event with the default strategies
    result = womens_team_pursuit.simulate(womens_transition_strategy, womens_pacing_strategy)

    print(result.get_finish_time())

    if result.get_finish_time() > 10000:  # in case of inf
        return 10000
    else:
        return float(result.get_finish_time())
