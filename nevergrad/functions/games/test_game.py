import typing as tp
from nevergrad.common import testing
import numpy as np
from . import game


@testing.parametrized(**{name: (name,) for name in game._Game().get_list_of_games()})
def test_games(name: str) -> None:
    dimension = game._Game().play_game(name)
    res: tp.List[tp.Any] = []
    for _ in range(200):
        res += [game._Game().play_game(name, np.random.uniform(0, 1, dimension), None)]
        score = (float(sum(1 if r == 2 else 0 if r == 1 else 0.5 for r in res)) / len(res))
    assert score >= 0.1
    assert score <= 0.9
    function = game.Game(name)
    function(function.instrumentation.random_state.normal(size=function.dimension))
