# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.common import testing
from . import game


@testing.parametrized(**{name: (name,) for name in game._Game().get_list_of_games()})
def test_games(name: str) -> None:
    dimension = game._Game().play_game(name)
    res: tp.List[tp.Any] = []
    for k in range(200):
        res.append(game._Game().play_game(name, np.random.uniform(0, 1, dimension), None))
        score = float(sum(1 if r == 2 else 0 if r == 1 else 0.5 for r in res)) / len(res)
        if k >= 20 and 0.2 <= score <= 0.8:
            break
    assert score >= 0.1
    assert score <= 0.9
    assert any(res), "All ties"
    function = game.Game(name)
    function(function.parametrization.random_state.normal(size=function.dimension))
