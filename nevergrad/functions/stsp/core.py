# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
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
from ... import instrumentation as inst
from ...instrumentation.multivariables import Instrumentation


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements,too-many-locals
class STSP(inst.InstrumentedFunction):

    def __init__(self, seed: int = 0, the_dimension: int = 500) -> None:
        instrumentation = Instrumentation(inst.var.Array(the_dimension))
        self.x = instrumentation.random_state.normal(size=the_dimension)
        self.y = instrumentation.random_state.normal(size=the_dimension)
        super().__init__(self._simulate_stsp, instrumentation)
        self._descriptors.update(seed=seed)
        self.order = np.arange(0, self.instrumentation.dimension)

    def _simulate_stsp(self, x: np.ndarray) -> float:
        order = np.argsort(x)
        self.order = order
        x = self.x[order]
        y = self.y[order]
        output = np.sqrt((x[0] - x[-1])**2 + (y[0] - y[-1])**2) + sum(np.sqrt((x[i] - x[i + 1])**2 + (y[i] - y[i + 1])**2)
                                                                      for i in range(self.dimension - 1))
        return float(output)

    def make_plots(self, filename: str = "stsp.png") -> None:
        plt.clf()
        # Plot the optimization run.
        ax = plt.subplot(1, 1, 1)
        ax.set_xlabel('iteration number')
        order = self.order
        x = self.x
        y = self.y
        ax.plot((x[o] for o in order), (y[o] for o in order))
        plt.savefig(filename)
