# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import utils


def test_transform() -> None:
    indices = list(range(12))
    transform = utils.Transform(indices, rotation=True)
    assert transform.rotation_matrix is not None
    rot: np.ndarray = transform.rotation_matrix
    np.testing.assert_array_almost_equal(rot.T.dot(rot), np.identity(12))
    x = np.random.normal(0, 1, 16)
    y = transform(x)
    np.testing.assert_equal(len(y), 12)
    np.testing.assert_array_equal(y, transform(x))
