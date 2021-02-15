# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np


class Transform:
    """Defines a unique random transformation (index selection, translation, and optionally rotation)
    which can be applied to a point
    """

    def __init__(
        self,
        indices: tp.List[int],
        translation_factor: float = 1,
        rotation: bool = False,
        random_state: np.random.RandomState = None,
    ) -> None:
        dim = len(indices)
        assert dim
        if random_state is None:
            random_state = np.random.RandomState(0)
            random_state.set_state(np.random.get_state())
        self.indices = np.asarray(indices)
        self.translation: np.ndarray = random_state.normal(0, 1, dim) * translation_factor
        self.rotation_matrix: tp.Optional[np.ndarray] = None
        if rotation:
            self.rotation_matrix = np.linalg.qr(random_state.normal(0, 1, size=(dim, dim)))[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = x[self.indices] - self.translation
        if self.rotation_matrix is not None:
            y = self.rotation_matrix.dot(y)
        return y
