# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp
import torch
import numpy as np
import imquality.brisque as brisque
import lpips
import cv2
from nevergrad.functions.base import UnsupportedExperiment as UnsupportedExperiment
from nevergrad.common.decorators import Registry


registry: Registry[tp.Any] = Registry()
MODELS: tp.Dict[str, tp.Any] = {}


class ImageLoss:

    REQUIRES_REFERENCE = True

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        if reference is not None:
            self.reference = reference
            assert len(self.reference.shape) == 3, self.reference.shape
            assert self.reference.min() >= 0.0
            assert self.reference.max() <= 256.0, f"Image max = {self.reference.max()}"
            assert self.reference.max() > 3.0  # Not totally sure but entirely black images are not very cool.
            self.domain_shape = self.reference.shape

    def __call__(self, img: np.ndarray) -> float:
        raise NotImplementedError(f"__call__ undefined in class {type(self)}")


@registry.register
class SumAbsoluteDifferences(ImageLoss):
    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        value = float(np.sum(np.fabs(x - self.reference)))
        return value


class Lpips(ImageLoss):
    def __init__(self, reference: tp.Optional[np.ndarray] = None, net: str = "") -> None:
        super().__init__(reference)
        self.net = net

    def __call__(self, img: np.ndarray) -> float:
        if self.net not in MODELS:
            MODELS[self.net] = lpips.LPIPS(net=self.net)
        loss_fn = MODELS[self.net]
        assert img.shape[2] == 3
        assert len(img.shape) == 3
        assert img.max() <= 256.0, f"Image max = {img.max()}"
        assert img.min() >= 0.0
        assert img.max() > 3.0
        img0 = torch.clamp(torch.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2) / 256.0, 0, 1) * 2.0 - 1.0
        img1 = (
            torch.clamp(torch.Tensor(self.reference.copy()).unsqueeze(0).permute(0, 3, 1, 2) / 256.0, 0, 1)
            * 2.0
            - 1.0
        )  # The copy operation is here because of a warning otherwise, as Torch does not support non-writable numpy arrays.
        return float(loss_fn(img0, img1))


@registry.register
class LpipsAlex(Lpips):
    def __init__(self, reference: np.ndarray) -> None:
        super().__init__(reference, net="alex")


@registry.register
class LpipsVgg(Lpips):
    def __init__(self, reference: np.ndarray) -> None:
        super().__init__(reference, net="vgg")


@registry.register
class SumSquareDifferences(ImageLoss):
    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        value = float(np.sum((x - self.reference) ** 2))
        return value


@registry.register
class HistogramDifference(ImageLoss):
    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        assert x.shape[2] == 3
        x_gray_1d = np.sum(x, 2).ravel()
        ref_gray_1d = np.sum(self.reference, 2).ravel()
        value = float(np.sum(np.sort(x_gray_1d) - np.sort(ref_gray_1d)))
        return value


@registry.register
class Koncept512(ImageLoss):
    """
    This loss uses the neural network Koncept512 to score images
    It takes one image or a list of images of shape [x, y, 3], with each pixel between 0 and 256, and returns a score.
    """

    REQUIRES_REFERENCE = False

    @property
    def koncept(self) -> tp.Any:  # cache the model
        key = "koncept"
        if key not in MODELS:
            if os.name != "nt":
                # pylint: disable=import-outside-toplevel
                from koncept.models import Koncept512 as K512Model

                MODELS[key] = K512Model()
            else:
                raise UnsupportedExperiment("Koncept512 is not working properly under Windows")
        return MODELS[key]

    def __call__(self, img: np.ndarray) -> float:
        loss = -self.koncept.assess(img)
        return float(loss)


@registry.register
class Blur(ImageLoss):
    """
    This estimates bluriness.
    """

    REQUIRES_REFERENCE = False

    def __call__(self, img: np.ndarray) -> float:
        assert img.shape[2] == 3
        assert len(img.shape) == 3
        img = np.asarray(img, dtype=np.float64)
        return -float(cv2.Laplacian(img, cv2.CV_64F).var())


@registry.register
class Brisque(ImageLoss):
    """
    This estimates the Brisque score (lower is better).
    """

    REQUIRES_REFERENCE = False

    def __call__(self, img: np.ndarray) -> float:
        try:
            score = brisque.score(img)
        except AssertionError:  # oh my god, brisque can raise an assert when the data is too weird.
            score = float("inf")
        return score
