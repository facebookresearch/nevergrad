import numpy as np
from koncept.models import Koncept512


class ImageLoss:
    def __init__(self, reference=None) -> None:
        self.reference = reference

    def __call__(self, img: np.ndarray) -> float:
        raise NotImplementedError


class SumAbsoluteDifferences(ImageLoss):
    def __init__(self, reference: np.ndarray) -> None:
        super().__init__(reference)
        self.domain_shape = self.reference.shape

    def __call__(self, x: np.ndarray) -> float:
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        value = float(np.linalg.norm(x - self.reference, 1))
        return value


class Koncept512(ImageLoss):
    """
    This loss uses the neural network Koncept512 to score images
    It takes one image or a list of images of shape [x, y, 3] and returns a score
    """
    def __init__(self) -> None:
        super().__init__()
        self.koncept = Koncept512()

    def __call__(self, img: np.ndarray) -> float:
        loss = self.koncept.assess(img)
        if len(loss.shape) == 0:
            loss = float(loss)
        return loss
