# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Based on a discussion at Dagstuhl's seminar on Computational Intelligence in Games with:
# - Dan Ashlock
# - Chiara Sironi
# - Guenter Rudolph
# - Jialin Liu

import matplotlib.pyplot as plt
import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction


class TO(ExperimentFunction):
    def __init__(self, n: int = 50) -> None:
        super().__init__(self._simulate_to, p.Array(shape=(n, n)))
        self.n = n
        self.idx = self.parametrization.random_state.randint(50000)

    def _simulate_to(self, x: np.ndarray) -> float:
        x = x.reshape(self.n, self.n)
        idx = self.idx
        xa = idx % 3
        xb = 2 - xa
        xs = 1.5 * (
            np.array(
                [float(np.cos(self.idx * 0.01 + xa * i + xb * j) < 0.0) for i in range(n) for j in range(n)]
            ).reshape(n, n)
            - 0.5
        )
        if (idx // 3) % 2 > 0:
            xs = np.transpose(xs)
        if (idx // 6) % 2 > 0:
            xs = -xs
        return (
            5.0 * np.sum(np.abs(x - xs) > 0.3) / size
            + 13.0 * np.linalg.norm(x - gaussian_filter(x, sigma=3)) / sqrtsize
        )
