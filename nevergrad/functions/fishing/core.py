# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/Foloso/MixSimulator/tree/nevergrad_experiment

import numpy as np
from nevergrad.parametrization import parameter as p
from .. import base


class OptimizeFish(base.ExperimentFunction):
    """
    Fishing simulator.

    Parameters
    ----------
    time: int
        number of days of the planning

    """

    def __init__(self, time: int = 365) -> None:
        init = 0.5 * np.ones(time)
        parameter = p.Array(init=init)
        parameter.set_bounds(0.0 * init, 2.0 * init)
        parameter.set_name("raw_encoding")
        super().__init__(_compute_total_fishing, parameter)


def _compute_total_fishing(list_number_fishermen: np.ndarray) -> float:
    """Lotka-Volterra equations.

    This computes the total fishing, given the fishing effort every day.
    The problem makes sense for abritrary number of days, so that this works for
    any length of the input. 365 means one year."""
    number_dogfishes = 0.2
    number_haddocks = 0.8
    total_fishing = 0.0

    for number_fishermen in list(list_number_fishermen):
        number_fishermen = max(0, number_fishermen)
        # Number of haddocks eaten_haddocks today.
        number_eaten_haddocks = number_dogfishes * number_haddocks * 4 / 30
        # Haddock growth.
        number_haddocks += number_haddocks * 2 / 30
        number_haddocks -= number_eaten_haddocks
        # Natural growth of dogfishes.
        number_dogfishes -= (1 / 30) * number_dogfishes
        number_dogfishes += (5 / 2) * number_eaten_haddocks
        # Number of captured dogfishes.
        capture = min((1 / 10) * number_fishermen, number_dogfishes)
        number_dogfishes -= capture
        assert number_dogfishes >= 0.0
        total_fishing += capture
        number_dogfishes = min(1.0, number_dogfishes)
        number_haddocks = min(1.0, number_haddocks)
    return -total_fishing
