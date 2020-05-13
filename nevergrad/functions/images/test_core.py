# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
import nevergrad as ng
from . import core
from torchvision.models import resnet50
import torch.nn as nn
import torch
from torchvision import models
from torchvision.models.resnet import model_urls


class Classifier(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.model = nn.Linear(image_size * image_size * 3,
                               10)  # models.resnet50(pretrained=True) #TODO modify as linear classifier

    def forward(self, x):
        return self.model(x.flatten(x.shape[0], -1))


def test_images_adversarial() -> None:
    image_size = 224
    classifier = Classifier(image_size)
    # classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
    image = torch.rand((3, image_size, image_size))

    epsilon = 0.05
    targeted = False
    label = 3
    func = core.ImageAdversarial(classifier=classifier, image=image, label=label,
                                 targeted=targeted, epsilon=epsilon)
    x = np.zeros((3, image_size, image_size))
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert value < np.inf
    other_func = func.copy()
    value = func(x)
    assert value < np.inf


def test_images() -> None:
    func = core.Image()
    x = 7 * np.fabs(np.random.normal(size=func.domain_shape))
    # data = func.parametrization.spawn_child().set_standardized_data(x.flatten()).value
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert value < np.inf
    other_func = func.copy()
    value = func(x)
    assert value < np.inf
