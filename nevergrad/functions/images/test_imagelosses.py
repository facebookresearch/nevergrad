# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from nevergrad.functions.images import imagelosses


def test_l1_loss() -> None:
    loss = imagelosses.SumAbsoluteDifferences(reference=np.zeros((3, 2)))
    assert loss(np.ones((3, 2))) == 6


def all_losses() -> None:
    for loss in [
        imagelosses.SumAbsoluteDifferences,
        imagelosses.Lpips_alex,
        imagelosses.Lpips_vgg,
        imagelosses.SumSquareDifferences,
        imagelosses.HistogramDifference,
        imagelosses.Koncept512,
        imagelosses.Blur
    ]:
        assert loss(reference=np.zeros((3, 2)))(np.ones(3, 2)) > 0.0
