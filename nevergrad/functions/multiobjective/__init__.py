# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from nevergrad.common import errors
import nevergrad.common.typing as tp


class MultiobjectiveFunction:
    """MultiobjectiveFunction is deprecated and is removed after v0.4.3 "
    because it is no more needed. You should just pass a multiobjective loss to "
    optimizer.tell.\nSee https://facebookresearch.github.io/nevergrad/"
    optimization.html#multiobjective-minimization-with-nevergrad\n",
    """

    def __init__(
        self,
        multiobjective_function: tp.Callable[..., tp.ArrayLike],
        upper_bounds: tp.Optional[tp.ArrayLike] = None,
    ) -> None:
        raise errors.NevergradDeprecationError(
            "MultiobjectiveFunction is deprecated and is removed after v0.4.3 "
            "because it is no more needed. You should just pass a multiobjective loss to "
            "optimizer.tell.\nSee https://facebookresearch.github.io/nevergrad/"
            "optimization.html#multiobjective-minimization-with-nevergrad\n",
        )
