# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
import nevergrad as ng
from . import _datalayers


def test_scalar_module() -> None:
    ref = ng.p.Scalar()
    x = ng.p.Scalar(10) % 4
    assert x.value == 2
    assert x.get_standardized_data(reference=ref)[0] == 10
    x.value = 1
    assert x.get_standardized_data(reference=ref)[0] == 9  # find the closest


def test_bound_module() -> None:
    ref = ng.p.Scalar()
    with pytest.raises(ng.errors.NevergradValueError):
        _datalayers.Bound(3, 8, method="arctan")(ref)
    x = _datalayers.Bound(-1, 8, method="arctan")(ref)
    x.set_standardized_data([100], reference=ref)
    np.testing.assert_almost_equal(x.value, 7.97135306)


def test_log_layer() -> None:
    ref = ng.p.Scalar()
    x = 2 ** ng.p.Scalar()
    assert x.value == 1
    x.value = 16
    assert x.get_standardized_data(reference=ref)[0] == 4  # find the closest
    assert x.value == 16


def test_add_layer() -> None:
    x = ng.p.Scalar() - 4.0
    y = 6 + x
    assert y.value == 2


def test_multiply_layer() -> None:
    ref = ng.p.Scalar()
    x = ng.p.Scalar(5) / 5
    assert x.value == 1
    x.value = 2
    assert x.get_standardized_data(reference=ref)[0] == 10


def test_power() -> None:
    ref = ng.p.Scalar()
    x = 3 / ng.p.Scalar(6)
    assert x.value == 0.5
    x.value = 0.25
    assert x.get_standardized_data(reference=ref)[0] == 12
