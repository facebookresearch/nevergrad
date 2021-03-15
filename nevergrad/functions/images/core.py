# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import itertools
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import torch.nn as nn
import torch
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as tr

import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.common import errors
from .. import base
from . import imagelosses

# pylint: disable=abstract-method


class Image(base.ExperimentFunction):
    def __init__(
        self,
        problem_name: str = "recovering",
        index: int = 0,
        loss: tp.Type[imagelosses.ImageLoss] = imagelosses.SumAbsoluteDifferences,
        with_pgan: bool = False,
        num_images: int = 1,
    ) -> None:
        """
        problem_name: the type of problem we are working on.
           recovering: we directly try to recover the target image.§
        index: the index of the problem, inside the problem type.
           For example, if problem_name is "recovering" and index == 0,
           we try to recover the face of O. Teytaud.
        """

        # Storing high level information.
        self.domain_shape = (226, 226, 3)
        self.problem_name = problem_name
        self.index = index
        self.with_pgan = with_pgan
        self.num_images = num_images

        # Storing data necessary for the problem at hand.
        assert problem_name == "recovering"  # For the moment we have only this one.
        assert index == 0  # For the moment only 1 target.
        # path = os.path.dirname(__file__) + "/headrgb_olivier.png"
        path = Path(__file__).with_name("headrgb_olivier.png")
        image = PIL.Image.open(path).resize((self.domain_shape[0], self.domain_shape[1]), PIL.Image.ANTIALIAS)
        self.data = np.asarray(image)[:, :, :3]  # 4th Channel is pointless here, only 255.
        # parametrization
        if not with_pgan:
            assert num_images == 1
            array = ng.p.Array(init=128 * np.ones(self.domain_shape), mutable_sigma=True)
            array.set_mutation(sigma=35)
            array.set_bounds(lower=0, upper=255.99, method="clipping", full_range_sampling=True)
            max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
            array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1), max_size=max_size)).set_name("")  # type: ignore
            super().__init__(loss(reference=self.data), array)
        else:
            self.pgan_model = torch.hub.load(
                "facebookresearch/pytorch_GAN_zoo:hub",
                "PGAN",
                model_name="celebAHQ-512",
                pretrained=True,
                useGPU=False,
            )
            self.domain_shape = (num_images, 512)  # type: ignore
            initial_noise = np.random.normal(size=self.domain_shape)
            self.initial = np.random.normal(size=(1, 512))
            self.target = np.random.normal(size=(1, 512))
            array = ng.p.Array(init=initial_noise, mutable_sigma=True)
            array.set_mutation(sigma=35.0)
            array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1))).set_name("")
            self._descriptors.pop("use_gpu", None)
            super().__init__(self._loss_with_pgan, array)

        assert self.multiobjective_upper_bounds is None
        self.add_descriptors(loss=loss.__class__.__name__)
        self.loss_function = loss(reference=self.data)

    def _generate_images(self, x: np.ndarray) -> np.ndarray:
        """ Generates images tensor of shape [nb_images, x, y, 3] with pixels between 0 and 255"""
        # pylint: disable=not-callable
        noise = torch.tensor(x.astype("float32"))
        return ((self.pgan_model.test(noise).clamp(min=-1, max=1) + 1) * 255.99 / 2).permute(0, 2, 3, 1).cpu().numpy()[:, :, :, [2, 1, 0]]  # type: ignore

    def interpolate(self, base_image: np.ndarray, target: np.ndarray, k: int, num_images: int) -> np.ndarray:
        if num_images == 1:
            return target
        coef1 = k / (num_images - 1)
        coef2 = (num_images - 1 - k) / (num_images - 1)
        return coef1 * base_image + coef2 * target

    def _loss_with_pgan(self, x: np.ndarray, export_string: str = "") -> float:
        loss = 0.0
        factor = 1 if self.num_images < 2 else 10  # Number of intermediate images.
        num_total_images = factor * self.num_images
        for i in range(num_total_images):
            base_i = i // factor
            # We generate num_images images. The last one is close to target, the first one is close to initial if num_images > 1.
            base_image = self.interpolate(self.initial, self.target, i, num_total_images)
            movability = 0.5  # If only one image, then we move by 0.5.
            if self.num_images > 1:
                movability = 4 * (
                    0.25 - (i / (num_total_images - 1) - 0.5) ** 2
                )  # 1 if i == num_total_images/2, 0 if 0 or num_images-1
            moving = (
                movability
                * np.sqrt(self.dimension)
                * np.expand_dims(x[base_i], 0)
                / (1e-10 + np.linalg.norm(x[base_i]))
            )
            base_image = moving if self.num_images == 1 else base_image + moving
            image = self._generate_images(base_image).squeeze(0)
            image = cv2.resize(image, dsize=(226, 226), interpolation=cv2.INTER_NEAREST)
            if export_string:
                cv2.imwrite(f"{export_string}_image{i}_{num_total_images}_{self.num_images}.jpg", image)
            assert image.shape == (226, 226, 3), f"{x.shape} != {(226, 226, 3)}"
            loss += self.loss_function(image)
        return loss

    def export_to_images(self, x: np.ndarray, export_string: str = "export"):
        self._loss_with_pgan(x, export_string=export_string)


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
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
    def __init__(
        self,
        classifier: nn.Module,
        image: torch.Tensor,
        label: int = 0,
        targeted: bool = False,
        epsilon: float = 0.05,
    ) -> None:
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

        array = ng.p.Array(
            init=np.zeros(self.image.shape),
            mutable_sigma=True,
        ).set_name("")
        array.set_mutation(sigma=self.epsilon / 10)
        array.set_bounds(lower=-self.epsilon, upper=self.epsilon, method="clipping", full_range_sampling=True)
        max_size = ng.p.Scalar(lower=1, upper=200).set_integer_casting()
        array.set_recombination(ng.p.mutation.Crossover(axis=(1, 2), max_size=max_size))  # type: ignore
        super().__init__(self._loss, array)

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

    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float:
        """Returns wether the attack worked or not"""
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        x = recommendations[0].value
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
            data_loader = torch.utils.DataLoader(
                ifolder, batch_size=1, shuffle=True, num_workers=8, pin_memory=True
            )
        else:
            raise ValueError(f"{folder} is not a valid folder.")
        for data, target in itertools.islice(data_loader, 0, 100):
            _, pred = torch.max(classifier(data), axis=1)
            if pred == target:
                func = cls(
                    classifier=classifier, image=data[0], label=int(target), targeted=False, epsilon=0.05
                )
                func.add_descriptors(**tags)
                yield func


class ImageFromPGAN(base.ExperimentFunction):
    """
    Creates face images using a GAN from pytorch GAN zoo trained on celebAHQ and optimizes the noise vector of the GAN

    Parameters
    ----------
    problem_name: str
        the type of problem we are working on.
    initial_noise: np.ndarray
        the initial noise of the GAN. It should be of dimension (1, 512). If None, it is defined randomly.
    use_gpu: bool
        whether to use gpus to compute the images
    loss: ImageLoss
        which loss to use for the images (default: Koncept512)
    mutable_sigma: bool
        whether the sigma should be mutable
    sigma: float
        standard deviation of the initial mutations
    """

    def __init__(
        self,
        initial_noise: tp.Optional[np.ndarray] = None,
        use_gpu: bool = False,
        loss: tp.Optional[imagelosses.ImageLoss] = None,
        mutable_sigma: bool = True,
        sigma: float = 35,
    ) -> None:
        if loss is None:
            loss = imagelosses.Koncept512()
        if not torch.cuda.is_available():
            use_gpu = False
        # Storing high level information..
        if os.environ.get("CIRCLECI", False):
            raise errors.UnsupportedExperiment("ImageFromPGAN is not well supported in CircleCI")
        self.pgan_model = torch.hub.load(
            "facebookresearch/pytorch_GAN_zoo:hub",
            "PGAN",
            model_name="celebAHQ-512",
            pretrained=True,
            useGPU=use_gpu,
        )

        self.domain_shape = (1, 512)
        if initial_noise is None:
            initial_noise = np.random.normal(size=self.domain_shape)
        assert initial_noise.shape == self.domain_shape, (
            f"The shape of the initial noise vector was {initial_noise.shape}, "
            f"it should be {self.domain_shape}"
        )

        array = ng.p.Array(init=initial_noise, mutable_sigma=mutable_sigma)
        # parametrization
        array.set_mutation(sigma=sigma)
        array.set_recombination(ng.p.mutation.Crossover(axis=(0, 1))).set_name("")

        super().__init__(self._loss, array)
        self.loss_function = loss
        self._descriptors.pop("use_gpu", None)

        self.add_descriptors(loss=loss.__class__.__name__)

    def _loss(self, x: np.ndarray) -> float:
        image = self._generate_images(x)
        loss = self.loss_function(image)
        return loss

    def _generate_images(self, x: np.ndarray) -> np.ndarray:
        """ Generates images tensor of shape [nb_images, x, y, 3] with pixels between 0 and 255"""
        # pylint: disable=not-callable
        noise = torch.tensor(x.astype("float32"))
        return ((self.pgan_model.test(noise).clamp(min=-1, max=1) + 1) * 255.99 / 2).permute(0, 2, 3, 1).cpu().numpy()  # type: ignore
