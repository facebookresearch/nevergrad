# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

import numpy as np
import PIL.Image
import torch.nn as nn
import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as tr

import nevergrad as ng
import nevergrad.common.typing as tp
from .. import base
# pylint: disable=abstract-method


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


# #### Adversarial attacks ##### #


class Normalize(nn.Module):

    def __init__(self, mean: tp.ArrayLike, std: tp.ArrayLike) -> None:
        super().__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]


class Resnet50(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.norm = Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        self.model = resnet50(pretrained=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.norm(x))


class TestClassifier(nn.Module):

    def __init__(self, image_size: int = 224) -> None:
        super().__init__()
        self.model = nn.Linear(image_size * image_size * 3, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.shape[0], -1))


# pylint: disable=too-many-arguments,too-many-instance-attributes
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

    def _loss(self, x: np.ndarray) -> float:
        output_adv = self._get_classifier_output(x)
        value = float(self.criterion(output_adv, self.label).item())
        return value * (1.0 if self.targeted else -1.0)

    def _get_classifier_output(self, x: np.ndarray) -> tp.Any:
        # call to the classifier given the input array
        y = torch.Tensor(x)
        image_adv = torch.clamp(self.image + y, 0, 1)
        image_adv = image_adv.view(1, 3, self.imsize, self.imsize)
        return self.classifier(image_adv)

    # pylint: disable=arguments-differ
    def evaluation_function(self, x: np.ndarray) -> float:  # type: ignore
        """Returns wether the attack worked or not
        """
        output_adv = self._get_classifier_output(x)
        _, pred = torch.max(output_adv, axis=1)
        actual = int(self.label)
        return float(pred == actual if self.targeted else pred != actual)

    @classmethod
    def make_folder_functions(
            cls,
            folder: tp.Optional[tp.PathLike],
            model: str = "resnet50",
    ) -> tp.Generator["ImageAdversarial", None, None]:
        """

        Parameters
        ----------
        folder: str or None
            folder to use for reference images. If None, 1 random image is created.
        model: str
            model name to use

        Yields
        ------
        ExperimentFunction
            an experiment function corresponding to 1 of the image of the provided folder dataset.
        """
        assert model in {"resnet50", "test"}
        tags = {"folder": "#FAKE#" if folder is None else Path(folder).name, "model": model}
        classifier: tp.Any = Resnet50() if model == "resnet50" else TestClassifier()
        imsize = 224
        transform = tr.Compose([tr.Resize(imsize), tr.CenterCrop(imsize), tr.ToTensor()])
        if folder is None:
            x = torch.zeros(1, 3, 224, 224)
            _, pred = torch.max(classifier(x), axis=1)
            data_loader: tp.Iterable[tp.Tuple[tp.Any, tp.Any]] = [(x, pred)]
        elif Path(folder).is_dir():
            ifolder = torchvision.datasets.ImageFolder(folder, transform)
            data_loader = torch.utils.DataLoader(ifolder, batch_size=1, shuffle=True,
                                                 num_workers=8, pin_memory=True)
        else:
            raise ValueError(f"{folder} is not a valid folder.")
        for data, target in data_loader:
            _, pred = torch.max(classifier(data), axis=1)
            if pred == target:
                func = cls._with_tag(tags=tags, classifier=classifier, image=data[0],
                                     label=int(target), targeted=False, epsilon=0.05)
                yield func

    @classmethod
    def _with_tag(
            cls,
            tags: tp.Dict[str, str],
            **kwargs: tp.Any,
    ) -> "ImageAdversarial":
        # generates an instance with a hack so that additional tags are propagated to copies
        func = cls(**kwargs)
        func.add_descriptors(**tags)
        func._initialization_func = cls._with_tag  # type: ignore
        assert func._initialization_kwargs is not None
        func._initialization_kwargs["tags"] = tags
        return func
