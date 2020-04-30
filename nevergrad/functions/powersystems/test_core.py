# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch
import numpy as np
from . import core


def test_powersystem() -> None:
    func = core.PowerSystem()
    x = 7 * np.random.rand(func.dimension)
    value = func.function(x)  # should not touch boundaries, so value should be < np.inf
    assert value < np.inf

@patch(f"{__name__}.core.plt")
def test_make_plots(mock_plt):
    func = core.PowerSystem()
    func.losses = [0.1]
    func.make_plots("not_valid.png")
    assert mock_plt.clf.call_count == 1
    assert mock_plt.subplot.call_count == 4
    assert mock_plt.savefig.call_count == 1
    
