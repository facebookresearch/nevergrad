# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import PIL.Image
import numpy as np
from nevergrad.functions.images import imagelosses


def test_l1_loss() -> None:
    loss = imagelosses.SumAbsoluteDifferences(124. * reference=np.ones((300, 400, 3)))
    assert loss(np.ones((3, 4, 3))) == 36.0


def test_consistency_losses_with_oteytaud() -> None:
    path = Path(__file__).with_name("headrgb_olivier.png")
    image = PIL.Image.open(path).resize((256, 256), PIL.Image.ANTIALIAS)
    data = np.asarray(image)[:, :, :3]  # 4th Channel is pointless here, only 255.
    for loss_name, loss_class in imagelosses.registry.items():
        loss = loss_class(reference=data)
        random_data = np.random.uniform(low=0.0, high=255.0, size=data.shape)
        loss_data = loss(data)
        loss_random_data = loss(random_data)
        assert loss_data < loss_random_data, f"Loss {loss_name} fails on oteytaud's photo."
