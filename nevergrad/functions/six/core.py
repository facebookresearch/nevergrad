# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://raw.githubusercontent.com/purdue-orbital/rocket-simulation/master/Simulation2.py


import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from .six import dimension as dimension
from .six import play_games as play_games


class SixGame(ExperimentFunction):
    def __init__(self, num_players: int = 5) -> None:
        self.num_players = num_players
        super().__init__(self._simulate_six, p.Array(shape=(dimension(num_players=num_players),)))

    def _simulate_six(self, x: np.ndarray) -> float:
        return play_games(x, self.num_players)[0]
