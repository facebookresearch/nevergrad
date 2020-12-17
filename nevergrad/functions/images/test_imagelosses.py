# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import PIL.Image
import numpy as np
import pytest
from nevergrad.functions.images import imagelosses


def test_l1_loss() -> None:
    loss = imagelosses.SumAbsoluteDifferences(reference=124.0 * np.ones((300, 400, 3)))
    assert loss(np.ones((300, 400, 3))) == 44280000.0


def test_image_losses_with_reference() -> None:
    assert (
        len([l for l in imagelosses.registry.values() if issubclass(l, imagelosses.ImageLossWithReference)])
        > 2
    )


@pytest.mark.parametrize("loss_name", imagelosses.registry)  # type: ignore
def test_consistency_losses_with_oteytaud(loss_name: str) -> None:
    path = Path(__file__).with_name("headrgb_olivier.png")
    image = PIL.Image.open(path).resize((256, 256), PIL.Image.ANTIALIAS)
    data = np.asarray(image)[:, :, :3]  # 4th Channel is pointless here, only 255.
    loss_class = imagelosses.registry[loss_name]
    try:
        loss = loss_class(reference=data)
    except imagelosses.UnsupportedExperiment as e:
        raise pytest.skip(str(e))
    random_data = np.random.uniform(low=0.0, high=255.0, size=data.shape)
    loss_data = loss(data)
    assert loss_data < 1000.0
    assert loss_data > -1000.0
    assert loss_data == loss(data)
    assert isinstance(loss_data, float)
    loss_random_data = loss(random_data)
    assert loss_random_data == loss(random_data)
    assert "Blur" in loss_name or loss_data < loss_random_data, f"Loss {loss_name} fails on oteytaud's photo."
