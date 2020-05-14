# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.common import testing
from . import mlfunctionlib


@testing.parametrized(
    decision_tree_depth=(
        dict(regressor="decision_tree_depth", data_dimension=1, dataset="artificial"),
        dict(depth=3),
        0.0012629517,
    ),
    decision_tree_artificial=(
        dict(regressor="decision_tree", data_dimension=12, dataset="artificial"),
        dict(depth=3, criterion="mse", min_samples_split=0.001),
        0.180305329,
    ),
    mlp_artificial=(
        dict(regressor="mlp", data_dimension=2, dataset="artificial"),
        dict(activation="relu", solver="adam", alpha=0.01, learning_rate="constant"),
        0.003852263392,
    ),
    choosing_regressor_perceptron=(
        dict(regressor="any", data_dimension=2, dataset="artificial"),
        dict(activation="relu", solver="adam", alpha=0.01, learning_rate="constant", depth=3,
             criterion="mse", min_samples_split=0.001, regressor="decision_tree"),
        0.0051153637,
    ),
    choosing_regressor_mlp=(
        dict(regressor="any", data_dimension=2, dataset="artificial"),
        dict(activation="relu", solver="adam", alpha=0.01, learning_rate="constant", depth=3,
             criterion="mse", min_samples_split=0.001, regressor="mlp"),
        0.0038522633,
    ),
    decision_tree_boston=(
        dict(regressor="decision_tree", data_dimension=None, dataset="boston"),
        dict(depth=5, criterion="mse", min_samples_split=0.001),
        33.188238558,
    ),
    decision_tree_diabetes=(
        dict(regressor="decision_tree", data_dimension=None, dataset="diabetes"),
        dict(depth=5, criterion="mse", min_samples_split=0.001),
        5302.64340105
    ),
    decision_tree_cos=(
        dict(regressor="decision_tree", data_dimension=3, dataset="artificialcos"),
        dict(depth=5, criterion="mse", min_samples_split=0.001),
        0.0004427973,
    ),
    decision_tree_square=(
        dict(regressor="decision_tree", data_dimension=3, dataset="artificialsquare"),
        dict(depth=5, criterion="mse", min_samples_split=0.001),
        0.001891651,
    ),
)
def test_mltuning_values(cls_params: tp.Dict[str, tp.Any], func_params: tp.Dict[str, tp.Any], expected: float) -> None:
    np.random.seed(12)
    func = mlfunctionlib.MLTuning(**cls_params)
    outputs = [func(**func_params) for _ in range(2)]
    assert outputs[0] == outputs[1]
    np.testing.assert_almost_equal(outputs[0], expected, decimal=8)
    # check that evaluation function is noisefree
    outputs_eval = [func(**func_params) for _ in range(2)]
    assert outputs_eval[0] == outputs_eval[1]
