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

model_urls['resnet50'] = model_urls['resnet50'].replace('https://', 'http://')


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.norm = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        self.model = models.resnet50(pretrained=True)

    def forward(self, x):
        return self.model(self.norm(x))


def test_images_adversarial() -> None:
    image_size = 224
    classifier = Classifier()
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
