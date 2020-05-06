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
import torchvision.transforms as transforms
import torchvision.datasets as datasets


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
        self.model = resnet50(pretrained=True)

    def forward(self, x):
        return self.model(self.norm(x))


def test_images_adversarial() -> None:
    image_size = 224
    classifier = Classifier()
    classifier = torch.nn.DataParallel(classifier, device_ids=range(torch.cuda.device_count()))
    image_folder = datasets.ImageFolder("/datasets01_101/imagenet_full_size/061417/val",
                                        transforms.Compose([transforms.Resize(image_size),
                                                            transforms.CenterCrop(image_size),
                                                            transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(image_folder,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=8,
                                              pin_memory=True)
    image, label = next(iter(data_loader))
    print(image.shape, label.shape)
    params_attack = {"model":"epsilon": 0.05, "targeted": False, "label": 3}
    func = core.Image(problem_name='adversarial', params=params_attack)
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

