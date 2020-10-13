import pytest
import numpy as np
from . import core


@pytest.mark.parametrize("fid", range(1, 24))  # type: ignore
def test_PBO(fid: int) -> None:
    func = core.PBOFunction(fid, 0, 16)
    x = func.parametrization.sample()
    value = func(x)
    assert isinstance(value, float), "All output of the iohprofiler-functions should be float"
    assert np.isfinite(value)


def test_W_model() -> None:
    func = core.WModelFunction()
    x = func.parametrization.sample()
    func2 = core.PBOFunction(1, 0, 16)
    assert func(x) == func2(x), "W-model with default setting should equal base_function"
    func = core.WModelFunction(base_function="LeadingOnes")
    x = func.parametrization.sample()
    func2 = core.PBOFunction(2, 0, 16)
    assert func(x) == func2(x), "W-model with default setting should equal base_function"
