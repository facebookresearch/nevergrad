import numpy as np
from nevergrad.common import errors
import nevergrad.common.typing as tp
from . import utils


class _ScalarCasting(utils.Layered):
    """Cast Array as a scalar"""

    def _get_value(self) -> float:
        out = super()._get_value()  # pulls from previous layer
        if not isinstance(out, np.ndarray) or not out.size == 1:
            raise errors.NevergradRuntimeError("Scalar casting can only be applied to size=1 Data parameters")
        integer = np.issubdtype(out.dtype, np.int)
        out = (int if integer else float)(out[0])
        return out  # type: ignore

    def _set_value(self, value: tp.Any) -> None:
        if not isinstance(value, (float, int, np.float, np.int)):
            raise TypeError(f"Received a {type(value)} in place of a scalar (float, int)")
        value = np.array([value], dtype=float)


class IntegerCasting(utils.Layered):
    """Cast Data as integer (or integer array)"""

    def _get_value(self) -> np.ndarray:
        return np.round(super()._get_value()).astype(int)
