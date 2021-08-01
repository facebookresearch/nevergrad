# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from unittest.mock import patch
import numpy as np
from . import core


def test_powersystem_small() -> None:
    np.random.seed(12)
    dams = 2
    func = core.PowerSystem(num_dams=dams, num_years=0.2)
    x = [np.random.rand(func.dimension // dams) for _ in range(dams)]
    value = func.function(*x)
    np.testing.assert_almost_equal(value, 4266.8177479)


@patch(f"{__name__}.core.plt")
def test_make_plots(mock_plt: tp.Any) -> None:
    func = core.PowerSystem()
    func.losses = [0.1]
    func.make_plots("not_valid.png")
    assert mock_plt.clf.call_count == 1
    assert mock_plt.subplot.call_count == 4
    assert mock_plt.savefig.call_count == 1
