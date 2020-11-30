from unittest import SkipTest
import pytest
import numpy as np
from . import core


@pytest.mark.parametrize("fid", range(1, 24))  # type: ignore
def test_PBO(fid: int) -> None:
    try:
        func = core.PBOFunction(fid, 0, 16)
    except ImportError:
        raise SkipTest("IOH is not installed")
    x = func.parametrization.sample()
    value = func(x.value)
    assert isinstance(value, float), "All output of the iohprofiler-functions should be float"
    assert np.isfinite(value)


@pytest.mark.parametrize("instrumentation", ["Softmax", "Ordered"])
def test_PBO_parameterization(instrumentation) -> None:
    try:
        func = core.PBOFunction(1, 0, 16, instrumentation=instrumentation)
    except ImportError:
        raise SkipTest("IOH is not installed")
    x = func.parametrization.sample()
    value = func(x.value)
    assert isinstance(value, float), "All output of the iohprofiler-functions should be float"
    assert np.isfinite(value)


def test_W_model() -> None:
    try:
        func = core.WModelFunction()
    except ImportError:
        raise SkipTest("IOH is not installed")
    x = func.parametrization.sample()
    func2 = core.PBOFunction(1, 0, 16)
    assert func(x.value) == func2(x.value), "W-model with default setting should equal base_function"
    func = core.WModelFunction(base_function="LeadingOnes")
    x = func.parametrization.sample()
    func2 = core.PBOFunction(2, 0, 16)
    assert func(x.value) == func2(x.value), "W-model with default setting should equal base_function"
