from . import game
import numpy as np
from typing import Any
from typing import List

np.random.seed()
for g in ["war", "flip", "batawaf", "guesswho", "bigguesswho"]:
    dimension = game._Game().play_game(g)
    res: List[Any] = []
    for _ in range(200):
        res += [game._Game().play_game(g, np.random.uniform(0, 0, dimension), None)]
        score = (float(sum(1 if r == 2 else 0 if r == 1 else 0.5 for r in res)) / len(res))
    assert score >= 0.1
    assert score <= 0.9


