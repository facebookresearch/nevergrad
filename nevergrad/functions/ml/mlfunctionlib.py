# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#import hashlib
#import itertools
import typing as tp
import numpy as np
from functools import partial

from nevergrad.parametrization import parameter as p
from nevergrad.common import tools
from nevergrad.common.typetools import ArrayLike
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore

from ..base import ExperimentFunction
from .. import utils
from .. import corefuncs



class MLTuning(ExperimentFunction):
    """Class for generating ML hyperparameter tuning problems.
    """

    # Example of ML problem.
    def _decision_tree_parametrization(self, depth: int, dimension: int, criterion: str, 
                                       min_samples_split: float, regressor: str, noise_free: bool):
        # 10-folds cross-validation
        num_data: int = 80
        result: float = 0.
        for cv in range(10):
            # All data.
            X_all = np.arange(0., 1., 1. / num_data)
            random_state = np.random.RandomState(17)
            random_state.shuffle(X_all)
            
            # Training set.
            X = X_all[np.arange(num_data) % 10 != cv]
            X = X.reshape(-1, dimension)
            y = np.sum(np.sin(X), axis=1).ravel()
            
            # Validation set or test set (noise_free is True for test set).
            X_test = X_all[np.arange(num_data) % 10 == cv]
            X_test = X_test.reshape(-1, dimension)

            if noise_free:
                X_test = np.arange(0., 1., 1000000)
                random_state.shuffle(X_test)
                X_test = X_test.reshape(-1, dimension)
            y_test = np.sum(np.sin(X_test), axis=1).ravel()
    
            assert isinstance(depth, int), f"depth has class {type(depth)} and value {depth}."
    
            # Fit regression model
            if regressor == "decision_tree":
                regr = DecisionTreeRegressor(max_depth=depth, criterion=criterion,
                                             min_samples_split=min_samples_split)
            elif regressor == "mlp":
                regr = MLPRegressor(max_depth=depth, criterion=criterion,
                                             min_samples_split=min_samples_split)
            regr.fit(np.asarray(X), np.asarray(y))
    
            # Predict
            pred_test = regr.predict(X_test)
            result += np.sum((y_test - pred_test)**2)
        return result / num_data

    def __init__(self, problem_type: str):
        self.problem_type = problem_type

        if problem_type == "1d_decision_tree_regression":
            # Only the depth
            parametrization = p.Instrumentation(depth=p.Scalar(lower=1, upper=1200).set_integer_casting())        
            super().__init__(partial(self._decision_tree_parametrization,
                                     noise_free=False, criterion="mse",
                                     min_samples_split=0.00001, dimension=1,
                                     regressor="decision_tree"), parametrization)
            self.evaluation_function = partial(self._decision_tree_parametrization,  # type: ignore
                                               noise_free=True, criterion="mse", 
                                               dimension=1, min_samples_split=0.00001,
                                               regressor="decision_tree")
        elif problem_type == "1d_decision_tree_regression_full":
            # Adding criterion{“mse”, “friedman_mse”, “mae”}
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),
                min_samples_split=p.Log(lower=0.0000001, upper=1),
                regressor="decision_tree",
            )        
            super().__init__(partial(self._decision_tree_parametrization, dimension=1, noise_free=False), parametrization)
            self.evaluation_function = partial(self._decision_tree_parametrization, dimension=1, criterion="mse",  # type: ignore
                                               min_samples_split=0.00001,
                                               regressor="decision_tree", noise_free=True)
        elif problem_type == "2d_decision_tree_regression_full":
            # Adding criterion{“mse”, “friedman_mse”, “mae”}
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),
                min_samples_split=p.Log(lower=0.0000001, upper=1)
            )        
            super().__init__(partial(self._decision_tree_parametrization, dimension=2, noise_free=False,
                                     regressor="decision_tree"), parametrization)
            self.evaluation_function = partial(self._decision_tree_parametrization, dimension=2,  # type: ignore
                                               regressor="decision_tree", noise_free=True,)
        elif problem_type == "3d_decision_tree_regression_full":
            # Adding criterion{“mse”, “friedman_mse”, “mae”}
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),
                min_samples_split=p.Log(lower=0.0000001, upper=1)
            )        
            super().__init__(partial(self._decision_tree_parametrization, dimension=3, 
                                     regressor="decision_tree", noise_free=False), parametrization)
            self.evaluation_function = partial(self._decision_tree_parametrization, dimension=3,  # type: ignore
                                               regressor="decision_tree", noise_free=True)
        else:
            assert False, f"Problem type {problem_type} undefined!"
        self.register_initialization(problem_type=problem_type)
