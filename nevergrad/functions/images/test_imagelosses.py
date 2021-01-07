# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import PIL.Image
import numpy as np
import pytest
from nevergrad.functions.images import imagelosses


def test_reference() -> None:
    assert imagelosses.SumAbsoluteDifferences.REQUIRES_REFERENCE
    assert not imagelosses.Blur.REQUIRES_REFERENCE
    assert not imagelosses.Koncept512.REQUIRES_REFERENCE
    assert not imagelosses.Brisque.REQUIRES_REFERENCE
    assert len([loss for loss in imagelosses.registry.values() if loss.REQUIRES_REFERENCE]) == 5
    assert len([loss for loss in imagelosses.registry.values() if not loss.REQUIRES_REFERENCE]) == 3


def test_l1_loss() -> None:
    loss = imagelosses.SumAbsoluteDifferences(reference=124.0 * np.ones((300, 400, 3)))
    assert loss(np.ones((300, 400, 3))) == 44280000.0


@pytest.mark.parametrize("loss_name", imagelosses.registry)  # type: ignore
def test_consistency_losses_with_oteytaud(loss_name: str) -> None:
    path = Path(__file__).with_name("headrgb_olivier.png")
    image = PIL.Image.open(path).resize((256, 256), PIL.Image.ANTIALIAS)
    data = np.asarray(image)[:, :, :3]  # 4th Channel is pointless here, only 255.

    data_flip = np.flip(data, 0).copy()  # Copy necessary as some nets do not support negative stride.
    loss_class = imagelosses.registry[loss_name]
    loss = loss_class(reference=data)
    random_data = np.random.uniform(low=0.0, high=255.0, size=data.shape)
    loss_data = loss(data)
    assert loss_data < 1000.0
    assert loss_data > -1000.0
    assert loss_data == loss(data)
    assert isinstance(loss_data, float)
    loss_random_data = loss(random_data)
    assert loss_random_data == loss(random_data)
    assert "Blur" in loss_name or loss_data < loss_random_data, f"Loss {loss_name} fails on oteytaud's photo."
    if loss.REQUIRES_REFERENCE:
        assert loss_data == 0.0
        if "Histogram" not in loss_name:
            assert loss_data < loss(data_flip)
        else:
            assert loss_data == loss(data_flip)
