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
    def _ml_parametrization(self,
                            depth: int,  # Parameters for regression trees.
                            dimension: int,
                            criterion: str, 
                            min_samples_split: float,
                            solver: str,  # Parameters for neural nets.
                            activation: str,
                            alpha: float,
                            learning_rate: str,
                            regressor: str,  # Choice of learner.
                            noise_free: bool):
        # 10-folds cross-validation
        num_data: int = 120
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
                X_test = np.arange(0., 1., 1. / 60000)
                random_state.shuffle(X_test)
                X_test = X_test.reshape(-1, dimension)
            y_test = np.sum(np.sin(X_test), axis=1).ravel()
    
            assert isinstance(depth, int), f"depth has class {type(depth)} and value {depth}."
    
            # Fit regression model
            if regressor == "decision_tree":
                regr = DecisionTreeRegressor(max_depth=depth, criterion=criterion,
                                             min_samples_split=min_samples_split, random_state=0)
            else:
                assert regressor == "mlp", f"unknown regressor {regressor}."
                regr = MLPRegressor(alpha=alpha, activation=activation, solver=solver,
                                    learning_rate=learning_rate, random_state=0)
            regr.fit(np.asarray(X), np.asarray(y))
    
            # Predict
            pred_test = regr.predict(X_test)
            result += np.sum((y_test - pred_test)**2)
        return result / num_data

    def __init__(self, regressor: str, dimension: int):
        self.regressor = regressor
        self.name = regressor + f"Dim{dimension}"

        if regressor == "decision_tree_depth":
            # Only the depth
            parametrization = p.Instrumentation(depth=p.Scalar(lower=1, upper=1200).set_integer_casting())        
            super().__init__(partial(self._ml_parametrization,
                                     noise_free=False, criterion="mse",
                                     min_samples_split=0.00001, dimension=dimension,
                                     regressor="decision_tree",
                                     alpha=1.0, learning_rate="no", 
                                     activation="no", solver="no"), parametrization)
            self.evaluation_function = partial(self._ml_parametrization,  # type: ignore
                                               noise_free=True, criterion="mse", 
                                               dimension=dimension, min_samples_split=0.00001,
                                               regressor="decision_tree",        
                                               alpha=1.0, learning_rate="no", 
                                               activation="no", solver="no")
        elif regressor == "any":
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),
                min_samples_split=p.Log(lower=0.0000001, upper=1),
                regressor=p.Choice(["mlp", "decision_tree"]),
                activation=p.Choice(["identity", "logistic", "tanh", "relu"]),
                solver=p.Choice(["lbfgs", "sgd", "adam"]),
                learning_rate=p.Choice(["constant", "invscaling", "adaptive"]),
                alpha=p.Log(lower=0.0000001, upper=1.),
            )        
            super().__init__(partial(self._ml_parametrization, dimension=dimension, noise_free=False), parametrization)
            self.evaluation_function = partial(self._ml_parametrization, dimension=dimension, noise_free=True)  # type: ignore
        elif regressor == "decision_tree":
            # Adding criterion{“mse”, “friedman_mse”, “mae”}
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),
                min_samples_split=p.Log(lower=0.0000001, upper=1),
                regressor="decision_tree",
            )        
            super().__init__(partial(self._ml_parametrization, dimension=dimension, noise_free=False,        
                                     alpha=1.0, learning_rate="no", regressor="decision_tree", 
                                     activation="no", solver="no"), parametrization)
            self.evaluation_function = partial(self._ml_parametrization, dimension=dimension, criterion="mse",  # type: ignore
                                               min_samples_split=0.00001,
                                               regressor="decision_tree", noise_free=True,        
                                               alpha=1.0, learning_rate="no", 
                                               activation="no", solver="no")
        elif regressor == "mlp":
            parametrization = p.Instrumentation(
                activation=p.Choice(["identity", "logistic", "tanh", "relu"]),
                solver=p.Choice(["lbfgs", "sgd", "adam"]),
                regressor="mlp",
                learning_rate=p.Choice(["constant", "invscaling", "adaptive"]),
                alpha=p.Log(lower=0.0000001, upper=1.),
            )        
            super().__init__(partial(self._ml_parametrization, dimension=dimension, noise_free=False,
                                     regressor="mlp", depth=-3, criterion="no", min_samples_split=0.1), parametrization)
            self.evaluation_function = partial(self._ml_parametrization, dimension=dimension,  # type: ignore
                                               regressor="mlp", noise_free=True, 
                                               depth=-3, criterion="no", min_samples_split=0.1)
        else:
            assert False, f"Problem type {regressor} undefined!"
        self.register_initialization(regressor=regressor, dimension=dimension)
