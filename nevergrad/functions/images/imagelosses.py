import numpy as np


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
        value = float(np.sum(np.fabs(x - self.reference)))
        return value


class Koncept512(ImageLoss):
    """
    This loss uses the neural network Koncept512 to score images
    It takes one image or a list of images of shape [x, y, 3] and returns a score
    """
    def __init__(self) -> None:
        super().__init__()
        import os
        if os.name != 'nt':
            from koncept.models import Koncept512 as K512_model  # type: ignore
            self.koncept = K512_model()
        else:
            self.koncept = None

    def __call__(self, img: np.ndarray) -> float:
        loss = self.koncept.assess(img) if self.koncept else np.zeros(1)
        if len(loss.shape) == 0:
            loss = float(loss)
        return loss
