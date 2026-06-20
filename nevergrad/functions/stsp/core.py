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


class STSP(ExperimentFunction):
    def __init__(self, dimension: int = 500, complex_tsp: bool = False) -> None:
        super().__init__(self._simulate_stsp, p.Array(shape=(dimension,)))
        self.order = np.arange(0, self.dimension)
        self.complex = complex_tsp
        self.x = self.parametrization.random_state.normal(size=self.dimension)
        self.y = self.parametrization.random_state.normal(size=self.dimension)

    def _simulate_stsp(self, x: np.ndarray) -> float:
        order = np.argsort(x)
        self.order = order
        x = self.x[order]
        y = self.y[order]
        output = np.sqrt((x[0] - x[-1]) ** 2 + (y[0] - y[-1]) ** 2) + sum(
            np.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) for i in range(self.dimension - 1)
        )
        if self.complex:
            output += np.sum(np.diff(order) ** 2) / np.sqrt(self.dimension + 1.0)
        return float(output)

    def make_plots(self, filename: str = "stsp.png") -> None:
        plt.clf()
        # Plot the optimization run.
        ax = plt.subplot(1, 1, 1)
        ax.set_xlabel("iteration number")
        order = self.order
        x = self.x
        y = self.y
        ax.plot((x[o] for o in order), (y[o] for o in order))  # type: ignore
        plt.savefig(filename)
