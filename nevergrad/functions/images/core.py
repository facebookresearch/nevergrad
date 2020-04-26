# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import sqrt, tan, pi
import numpy as np
import nevergrad as ng
from nevergrad.common.typetools import ArrayLike
from .. import base


class Images(base.ExperimentFunction):
    def __init__(self) -> None:
        init = TODO
        array = ng.p.Array(init=init, mutable_sigma=True,)
        array.set_mutation(sigma=sigma)
        array.set_bounds(self.epmin, self.epf, method=bounding_method, full_range_sampling=True)
        array.set_recombination(ng.p.mutation.Crossover(0)).set_name("")
        super().__init__(self._loss, array)
        self.register_initialization()
        self._descriptors.update()

    def _loss(self, x: np.ndarray) -> float:
        x = np.array(x, copy=False).ravel()
        assert len(x) == self.dimension, f"Expected dimension {self.dimension}, got {len(x)}"
        value = TODO
        return value

    # pylint: disable=arguments-differ
    def evaluation_function(self, x: np.ndarray) -> float:  # type: ignore
        loss = self.function(x)
        return loss
