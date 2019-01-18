# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Tuple, Optional
import numpy as np
from . import base


@base.registry.register
class CustomOptimizer(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sigma: float = 1

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        else:
            return self.current_bests["pessimistic"].x + self.sigma * np.random.normal(0, 1, self.dimension)

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        if value <= self.current_bests["pessimistic"].mean:
            self.sigma = 2. * self.sigma
        else:
            self.sigma = .84 * self.sigma

    def _internal_provide_recommendation(self) -> Tuple[float, ...]:
        return self.current_bests["pessimistic"].x
