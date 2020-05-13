# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import sqrt, tan, pi
from pathlib import Path

import numpy as np
import nevergrad as ng
import PIL.Image
import os
from nevergrad.common.typetools import ArrayLike
from .. import base
import torch.nn as nn
import torch
from torchvision.models import resnet50


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        self.model = resnet50(pretrained=True)

    def forward(self, x):
        return self.model(self.norm(x))


class Image(base.ExperimentFunction):
    def __init__(self, problem_name: str = "recovering", index: int = 0) -> None:
        """
        problem_name: the type of problem we are working on.
           recovering: we directly try to recover the target image.
        index: the index of the problem, inside the problem type.
           For example, if problem_name is "recovering" and index == 0,
           we try to recover the face of O. Teytaud.
        """

        # Storing high level information.
        self.domain_shape = (256, 256, 3)
        self.problem_name = problem_name
        self.index = index

        # Storing data necessary for the problem at hand.
        assert problem_name == "recovering"  # For the moment we have only this one.
        assert index == 0  # For the moment only 1 target.
        # path = os.path.dirname(__file__) + "/headrgb_olivier.png"
        path = Path(__file__).with_name("headrgb_olivier.png")
        image = PIL.Image.open(path).resize((self.domain_shape[0], self.domain_shape[1]), PIL.Image.ANTIALIAS)
        self.data = np.asarray(image)[:, :, :3]  # 4th Channel is pointless here, only 255.

        array = ng.p.Array(init=128 * np.ones(self.domain_shape), mutable_sigma=True, )
        array.set_mutation(sigma=35)
        array.set_bounds(lower=0, upper=255.99, method="clipping", full_range_sampling=True)
        max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1), max_size=max_size)).set_name("")  # type: ignore

        super().__init__(self._loss, array)
        self.register_initialization(problem_name=problem_name, index=index)
        self._descriptors.update(problem_name=problem_name, index=index)

    def _loss(self, x: np.ndarray) -> float:
        assert self.problem_name == "recovering"
        x = np.array(x, copy=False).ravel()
        x = x.reshape(self.domain_shape)
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        # Define the loss, in case of recovering: the goal is to find the target image.
        assert self.index == 0
        value = np.sum(np.fabs(np.subtract(x, self.data)))
        return value


class ImageAdversarial(base.ExperimentFunction):
    def __init__(self, classifier: nn.Module = None, image: torch.Tensor = None, label: int = None,
                 targeted: bool = False, epsilon: float = 0.05) -> None:
        # TODO add crossover params in args + criterion
        """
        params : needs to be detailed
        """
        self.targeted = targeted
        self.epsilon = epsilon
        self.image = image if (image is not None) else torch.rand((3, 224, 224))
        self.image_size = self.image.shape[1]
        self.domain_shape = self.image.shape  # (3,self.image_size, self.image_size)
        self.label = torch.Tensor([label]) if (label is not None) else torch.Tensor([0])
        self.label = self.label.long()
        self.classifier = classifier if (classifier is not None) else Classifier()
        self.criterion = nn.CrossEntropyLoss()

        array = ng.p.Array(init=np.zeros(self.domain_shape), mutable_sigma=True, )
        array.set_mutation(sigma=self.epsilon / 10)
        array.set_bounds(lower=-self.epsilon, upper=self.epsilon, method="clipping", full_range_sampling=True)
        max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array.set_recombination(ng.p.mutation.Crossover(axis=(1, 2),
                                                        max_size=max_size)).set_name("")  # type: ignore

        super().__init__(self._loss, array)
        self.register_initialization(classifier=classifier, image=image, label=label,
                                     targeted=targeted, epsilon=epsilon)
        self._descriptors.update(classifier=classifier, image=image, label=label,
                                 targeted=targeted, epsilon=epsilon)

    def _loss(self, x: np.ndarray) -> float:
        x = torch.Tensor(x)
        image_adv = torch.clamp(self.image + x, 0, 1)
        image_adv = image_adv.view(1, 3, self.image_size, self.image_size)
        output_adv = self.classifier(image_adv)
        if self.targeted:
            value = self.criterion(output_adv, self.label)
        else:
            value = -self.criterion(output_adv, self.label)
        value = value.item()
        return value
