# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from unittest import SkipTest
import pytest
import numpy as np
from nevergrad.optimization.optimizerlib import OnePlusOne
from . import core


@pytest.mark.parametrize("fid", range(1, 24))  # type: ignore
@pytest.mark.parametrize("iid", range(1, 6))  # type: ignore
def test_PBO(fid: int, iid: int) -> None:
    values = []
    try:
        func = core.PBOFunction(fid, iid, 16)
    except ModuleNotFoundError as e:
        raise SkipTest("IOH is not installed") from e
    for _ in range(30):
        x = func.parametrization.sample()
        assert not set(x.value) - {0, 1}, f"Non binary sample {x}."
        value = func(x.value)
        assert isinstance(value, float), "All output of the iohprofiler-functions should be float"
        assert np.isfinite(value)
        values.append(value)
    optim = OnePlusOne(func.parametrization, budget=100)
    recom = optim.minimize(func)
    values.append(func(recom.value))  # type: ignore
    assert (
        fid in [19, 20, 21, 22, 23] or min(values) >= 0.0 or max(values) <= 0.0
    ), f"IOH profile functions should have constant sign: pb with fid={fid},iid={iid}."


@pytest.mark.parametrize("instrumentation", ["Softmax", "Ordered"])  # type: ignore
def test_PBO_parameterization(instrumentation: str) -> None:
    try:
        func = core.PBOFunction(1, 0, 16, instrumentation=instrumentation)
    except ModuleNotFoundError as e:
        raise SkipTest("IOH is not installed") from e
    x = func.parametrization.sample()
    value = func(x.value)
    assert isinstance(value, float), "All output of the iohprofiler-functions should be float"
    assert np.isfinite(value)


def test_W_model() -> None:
    try:
        func = core.WModelFunction()
    except ModuleNotFoundError as e:
        raise SkipTest("IOH is not installed") from e
    x = func.parametrization.sample()
    func2 = core.PBOFunction(1, 0, 16)
    assert func(x.value) == func2(x.value), "W-model with default setting should equal base_function"
    func = core.WModelFunction(base_function="LeadingOnes")
    x = func.parametrization.sample()
    func2 = core.PBOFunction(2, 0, 16)
    assert func(x.value) == func2(x.value), "W-model with default setting should equal base_function"


_X = [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1]


# reference values were computed with the former IOHexperimenter 0.2.9.2 backend
# (where MIS was available as fid 19 instead of 22)
@pytest.mark.parametrize(  # type: ignore
    "fid,iid,expected",
    [
        (1, 1, -10.0),
        (1, 3, -836.4213657234893),
        (3, 1, -84.0),
        (7, 3, -831.9113703459087),
        (14, 1, 0.0),
        (18, 3, -802.1061835027671),
        (22, 1, 150.0),
        (23, 3, -547.7816615583288),
    ],
)
def test_PBO_values(fid: int, iid: int, expected: float) -> None:
    try:
        func = core.PBOFunction(fid, iid, 16)
    except ModuleNotFoundError as e:
        raise SkipTest("IOH is not installed") from e
    assert func(_X) == pytest.approx(expected, rel=1e-12)


# reference values were computed with the former IOHexperimenter 0.2.9.2 backend
@pytest.mark.parametrize(  # type: ignore
    "kwargs,expected",
    [
        (dict(base_function="OneMax", iid=1, dummy=0.5), -7.0),
        (dict(base_function="LeadingOnes", iid=1, neutrality=3), -3.0),
        (dict(base_function="OneMax", iid=1, ruggedness=16), -11.0),
        (dict(base_function="LeadingOnes", iid=0, ruggedness=16), -3.0),
    ],
)
def test_W_model_values(kwargs: tp.Dict[str, tp.Any], expected: float) -> None:
    try:
        func = core.WModelFunction(dim=16, **kwargs)
    except ModuleNotFoundError as e:
        raise SkipTest("IOH is not installed") from e
    assert func(_X) == expected


def test_W_model_ruggedness_multiple_of_dim() -> None:
    try:
        import ioh  # noqa  # pylint: disable=unused-import,import-outside-toplevel
    except ModuleNotFoundError as e:
        raise SkipTest("IOH is not installed") from e
    with pytest.raises(ValueError, match="multiple of the dimension"):
        core.WModelFunction(dim=16, ruggedness=5)
