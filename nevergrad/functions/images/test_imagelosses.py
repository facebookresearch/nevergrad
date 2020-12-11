# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from nevergrad.functions.images import imagelosses


def test_l1_loss() -> None:
    loss = imagelosses.SumAbsoluteDifferences(reference=np.zeros((3, 2)))
    assert loss(np.ones((3, 2))) == 6
