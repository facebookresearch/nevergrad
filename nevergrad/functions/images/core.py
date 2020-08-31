# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import PIL.Image
import torch.nn as nn
import torch

import nevergrad as ng
from .. import base


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
        # parametrization
        array = ng.p.Array(init=128 * np.ones(self.domain_shape), mutable_sigma=True)
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
        value = float(np.sum(np.fabs(np.subtract(x, self.data))))
        return value


class TestClassifier(nn.Module):
    def __init__(self, image_size: int = 224):
        super().__init__()
        self.model = nn.Linear(image_size * image_size * 3, 10)

    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))


# pylint: disable=too-many-arguments
class ImageAdversarial(base.ExperimentFunction):

    def __init__(self, classifier: nn.Module, image: torch.Tensor, label: int = 0, targeted: bool = False,
                 epsilon: float = 0.05) -> None:
        # TODO add crossover params in args + criterion
        """
        params : needs to be detailed
        """
        self.targeted = targeted
        self.epsilon = epsilon
        self.image = image  # if (image is not None) else torch.rand((3, 224, 224))
        self.label = torch.Tensor([label])  # if (label is not None) else torch.Tensor([0])
        self.label = self.label.long()
        self.classifier = classifier  # if (classifier is not None) else Classifier()
        self.criterion = nn.CrossEntropyLoss()
        self.imsize = self.image.shape[1]

        array = ng.p.Array(init=np.zeros(self.image.shape), mutable_sigma=True, ).set_name("")
        array.set_mutation(sigma=self.epsilon / 10)
        array.set_bounds(lower=-self.epsilon, upper=self.epsilon, method="clipping", full_range_sampling=True)
        max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array.set_recombination(ng.p.mutation.Crossover(axis=(1, 2), max_size=max_size))  # type: ignore

        super().__init__(self._loss, array)
        self.register_initialization(classifier=classifier, image=image, label=label,
                                     targeted=targeted, epsilon=epsilon)
        # classifier and image cant be set as descriptors
        self.add_descriptors(label=label, targeted=targeted, epsilon=epsilon)

    @classmethod
    def from_testbed(
            cls,
            name: str,
            label: int = 0,
            targeted: bool = False,
            epsilon: float = 0.05
    ) -> "ImageAdversarial":
        if name == "test":
            imsize = 224
            classifier = TestClassifier(imsize)
            image = torch.rand((3, imsize, imsize))
        else:
            raise ValueError(f'Testbed "{name}" is not implemented, check implementation in {__file__}')
        func = cls(classifier=classifier, image=image, label=label, targeted=targeted, epsilon=epsilon)
        # clean up and update decsriptors
        assert func._initialization_kwargs is not None
        for d in ["classifier", "image"]:
            del func._initialization_kwargs[d]
        func._initialization_kwargs["name"] = name
        func._initialization_func = cls.from_testbed  # type: ignore
        func._descriptors.update(name=name)
        return func

    def _loss(self, x: np.ndarray) -> float:
        x = torch.Tensor(x)
        image_adv = torch.clamp(self.image + x, 0, 1)
        image_adv = image_adv.view(1, 3, self.imsize, self.imsize)
        output_adv = self.classifier(image_adv)
        if self.targeted:
            value = self.criterion(output_adv, self.label)
        else:
            value = -self.criterion(output_adv, self.label)
        return float(value.item())
