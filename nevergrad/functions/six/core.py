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
    """Game of 6nimmt.

    Parameters:
       num_players : int
           the number of players per game.
       config: int
           the index, in {0, 1, 2, 3, 4}, of the set of players.
    """
    def __init__(self, num_players: int = 5, config: int = 0) -> None:
        self.num_players = num_players
        self.config = config
        dim = dimension(num_players=num_players)
        super().__init__(
            self._simulate_six, p.Tuple(p.Array(shape=(dim[0],)), p.Array(shape=(dim[1],))).set_name("tuple")
        )
        assert self.dimension == sum(dim)

    def _simulate_six(self, x: np.ndarray) -> float:
        return play_games(x, num_players=self.num_players, config=self.config)[0]
