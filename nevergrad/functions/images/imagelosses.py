import cv2
import lpips
import os
import torch
import typing as tp
import numpy as np
from nevergrad.functions.base import UnsupportedExperiment
from nevergrad.common.decorators import Registry


registry: Registry[tp.Any] = Registry()


class ImageLoss:

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        pass

    def __call__(self, img: np.ndarray) -> float:
        raise NotImplementedError(f"__call__ undefined in class {type(self)}")


class ImageLossWithReference(ImageLoss):

    def __init__(self, reference: np.ndarray) -> None:
        if reference is None:
            raise ValueError("A reference is required")
        self.reference = reference
        super().__init__(reference)
        assert len(self.reference.shape) == 3, self.reference.shape
        self.domain_shape = self.reference.shape

@registry.register
class SumAbsoluteDifferences(ImageLossWithReference):

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__(reference)

    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        value = float(np.sum(np.fabs(x - self.reference)))
        return value


@registry.register
class LpipsAlex(ImageLossWithReference):

    def __init__(self, reference: np.ndarray) -> None:
        super().__init__(reference)
        self.loss_fn = lpips.LPIPS(net="alex")

    def __call__(self, img: np.ndarray) -> float:
        img0 = torch.clamp(torch.Tensor(img), 0, 1) * 2.0 - 1.0
        img1 = torch.clamp(torch.Tensor(self.reference), 0, 1) * 2.0 - 1.0
        assert len(img0.shape) == 4 and img0.shape[0] == 1
        assert len(img1.shape) == 4 and img1.shape[0] == 1
        assert all(np.fabs(img0.ravel()) <= 1.0)
        assert all(np.fabs(img1.ravel()) <= 1.0)
        return self.loss_fn(img0, img1)


@registry.register
class LpipsVgg(LpipsAlex):
    def __init__(self, reference: np.ndarray) -> None:
        super().__init__(reference)
        self.loss_fn = lpips.LPIPS(net="vgg")


@registry.register
class SumSquareDifferences(ImageLossWithReference):

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__(reference)

    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        value = float(np.sum((x - self.reference) ** 2))
        return value


@registry.register
class HistogramDifference(ImageLossWithReference):

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__(reference)

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
    It takes one image or a list of images of shape [x, y, 3], with each pixel between 0 and 255, and returns a score.
    """

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__()  # reference is useless in this case
        if os.name != "nt":
            # pylint: disable=import-outside-toplevel
            from koncept.models import Koncept512 as K512Model

            self.koncept = K512Model()
        else:
            raise UnsupportedExperiment("Koncept512 is not working properly under Windows")

    def __call__(self, img: np.ndarray) -> float:
        loss = - self.koncept.assess(img)
        return float(loss)


@registry.register
class Blur(ImageLoss):
    """
    This estimates bluriness
    """

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__()  # No reference needed! Just blur estimation.

    def __call__(self, img: np.ndarray) -> float:
        return cv2.Laplacian(image, cv2.CV_64F).var()


@registry.register
class NegBrisque(ImageLoss):
    """
    This estimates the negated Brisque score.
    """
    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__()  # No reference needed! Just Brisque estimation.

    def __call__(self, img: np.ndarray) -> float:
        return brisque.score(img)

