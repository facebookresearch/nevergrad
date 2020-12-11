import os
import typing as tp
import numpy as np
from nevergrad.functions.base import UnsupportedExperiment


class ImageLoss:
    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        pass

    def __call__(self, img: np.ndarray) -> float:
        raise NotImplementedError


class SumAbsoluteDifferences(ImageLoss):
    def __init__(self, reference: np.ndarray) -> None:
        if reference is None:
            raise ValueError("A reference is required")
        self.reference = reference
        super().__init__(reference)
        self.domain_shape = self.reference.shape

    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        value = float(np.sum(np.fabs(x - self.reference)))
        return value


class Koncept512(ImageLoss):
    """
    This loss uses the neural network Koncept512 to score images
    It takes one image or a list of images of shape [x, y, 3] and returns a score
    """

    def __init__(self, reference: tp.Optional[np.ndarray] = None) -> None:
        super().__init__()  # reference is useless in this case
        if os.name != 'nt':
            # pylint: disable=import-outside-toplevel
            from koncept.models import Koncept512 as K512Model
            self.koncept = K512Model()
        else:
            raise UnsupportedExperiment("Koncept512 is not working properly under Windows")

    def __call__(self, img: np.ndarray) -> float:
        loss = self.koncept.assess(img)
        return float(loss)
