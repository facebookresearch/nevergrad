import numpy as np
from koncept.models import Koncept512


class _ImageLoss:
    def __init__(self) -> None:
        pass

    def compute_loss(self, img: np.ndarray) -> float:
        assert False


class _ImageLossUsingRef(_ImageLoss):
    def __init__(self, reference: np.ndarray) -> None:
        super().__init__()
        self.reference = reference


class sumAbsoluteDifferencesLoss(_ImageLossUsingRef):
    def __init__(self, reference: np.ndarray) -> None:
        super().__init__(reference)
        self.domain_shape = self.reference.shape

    def compute_loss(self, x: np.ndarray) -> float:
        x = np.array(x, copy=False).ravel()
        x = x.reshape(self.domain_shape)
        assert x.shape == self.domain_shape, f"Shape = {x.shape} vs {self.domain_shape}"
        # Define the loss, in case of recovering: the goal is to find the target image.
        value = float(np.sum(np.fabs(np.subtract(x, self.reference))))
        return value


class Koncept512Loss(_ImageLoss):
    """
    This loss uses the neural network Koncept512 to score images
    It takes one image or a list of images of shape [x, y, 3] and returns a score
    """
    def __init__(self) -> None:
        super().__init__()
        self.koncept = Koncept512()

    def compute_loss(self, img: np.ndarray) -> float:
        loss = self.koncept.assess(img)
        if len(loss.shape) == 0:
            loss = float(loss)
        return loss
