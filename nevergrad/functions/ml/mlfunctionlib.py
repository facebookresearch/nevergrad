# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#import hashlib
#import itertools
import typing as tp
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools
from nevergrad.common.typetools import ArrayLike
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from ..base import ExperimentFunction
from . import utils
from . import corefuncs


# Example of ML problem.
def _decision_tree_parametrization(depth: int):
    # 10-folds cross-validation
    num_data: int = 80
    result: float = 0.
    for cv in range(10):
        X_all = np.arange(0., 1., 1. / num_data)
        X = X_all[np.arange(num_data) % 10 != cv]
        X_test = X_all[np.arange(num_data) % 10 == cv]

        for i in range(num_data):
            if i % 10 != cv:
                list_X += [[float(i)]]
            else:
                list_X_test += [[float(i)]]

        # Convert to numpy arrays.
        X = np.array([np.array(xi) for xi in list_X])
        X_test = np.array([np.array(xi) for xi in list_X_test])
        y = np.sin(X).ravel()

        assert isinstance(depth, int), f"depth has class {type(depth)} and value {depth}."

        # Fit regression model
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(np.asarray(X), np.asarray(y))

        # Predict
        pred_test = regr.predict(X_test)
        y_test = np.sin(np.asarray(X_test)).ravel()
        result += np.sum((y_test - pred_test)**2)
    return result / num_data


class MLTuning(ExperimentFunction):
    """Class for generating ML hyperparameter tuning problems.
    """
    def __init__(self, problem_type: str):
        self.name = problem_type
        self._parameters = {x: y for x, y in locals().items() if x not in ["__class__", "self"]}

        if problem_type == "1d_decision_tree_regression":
            parametrization = p.Instrumentation(depth=p.Scalar(lower=1, upper=1200).set_integer_casting())        
            super().__init__(_decision_tree_parametrization, parametrization)
            self.register_initialization(**self._parameters)
        else:
            assert False, f"Problem type {problem_type} undefined!"
