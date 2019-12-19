
from typing import Any
from typing import List
import numpy as np
from . import game

@pytest.mark.parametrize("name", ["war", "flip", "batawaf", "guesswho", "bigguesswho"])
def test_games(name: str) -> None
    np.random.seed()  # TODO REMOVE THIS, IT SHOULD NEVER BE NEEDED
    dimension = game._Game().play_game(g)
    res: List[Any] = []
    for _ in range(200):
        res += [game._Game().play_game(g, np.random.uniform(0, 0, dimension), None)]
        score = (float(sum(1 if r == 2 else 0 if r == 1 else 0.5 for r in res)) / len(res))
    assert score >= 0.1
    assert score <= 0.9


