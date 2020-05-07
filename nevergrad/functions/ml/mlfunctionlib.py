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

    def get_dataset(self, dimension):

        num_data: int = 120  # Training set size.
        self.num_data = num_data
        
        # Training set.
        X = np.arange(0., 1., 1. / (num_data * dimension))
        X = X.reshape(-1, dimension)
        random_state = np.random.RandomState(17)
        random_state.shuffle(X)
        y = np.sum(np.sin(X), axis=1).ravel()
        self.X = X  # Training set.
        self.y = y  # Labels of the training set.

        # Variables for storing the cross-validation splits.
        self.X_train = []  # This will be the list of training subsets.
        self.X_valid = []  # This will be the list of validation subsets.
        self.y_train = []
        self.y_valid = []

        # We generate the cross-validation subsets.
        for cv in range(10):

            # Training set.
            X_train = X[np.arange(num_data) % 10 != cv].copy()
            y_train = np.sum(np.sin(X_train), axis=1).ravel()
            self.X_train += [X_train]
            self.y_train += [y_train]

            # Validation set or test set (noise_free is True for test set).
            X_valid = X[np.arange(num_data) % 10 == cv].copy()
            X_valid = X_valid.reshape(-1, dimension)
            y_valid = np.sum(np.sin(X_valid), axis=1).ravel()
            self.X_valid += [X_valid]
            self.y_valid += [y_valid]

        # We also generate the test set.
        X_test = np.arange(0., 1., 1. / 60000)
        random_state.shuffle(X_test)
        X_test = X_test.reshape(-1, dimension)
        y_test = np.sum(np.sin(X_test), axis=1).ravel()
        self.X_test = X_test
        self.y_test = y_test

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
                            noise_free: bool  # Whether we work on the test set (the real cost) on an approximation (CV error on train).
                            ):
        # 10-folds cross-validation
        result: float = 0.
        # Fit regression model
        if regressor == "decision_tree":
            regr = DecisionTreeRegressor(max_depth=depth, criterion=criterion,
                                         min_samples_split=min_samples_split, random_state=0)
        else:
            assert regressor == "mlp", f"unknown regressor {regressor}."
            regr = MLPRegressor(alpha=alpha, activation=activation, solver=solver,
                                learning_rate=learning_rate, random_state=0)

        if noise_free:  # noise_free is True when we want the result on the test set.
            X = self.X
            y = self.y
            X_test = self.X_test
            y_test = self.y_test
            regr.fit(np.asarray(self.X), np.asarray(self.y))
            pred_test = regr.predict(self.X_test)
            return np.sum((self.y_test - pred_test)**2)

        # We do a cross-validation.
        for cv in range(10):

            X = self.X_train[cv]
            y = self.y_train[cv]
            X_test = self.X_valid[cv]
            y_test = self.y_valid[cv]

            assert isinstance(depth, int), f"depth has class {type(depth)} and value {depth}."
    
            regr.fit(np.asarray(X), np.asarray(y))
    
            # Predict
            pred_test = regr.predict(X_test)
            result += np.sum((y_test - pred_test)**2)

        return result / self.num_data  # We return a 10-fold validation error.

    def __init__(self, regressor: str, dimension: int):
        """We propose different possible regressors and different dimensionalities.
        In each case, Nevergrad will optimize the parameters of a scikit learning.
        """
        self.regressor = regressor
        self.name = regressor + f"Dim{dimension}"
        self.get_dataset(dimension)

        if regressor == "decision_tree_depth":
            # Only the depth, as an evaluation.
            parametrization = p.Instrumentation(depth=p.Scalar(lower=1, upper=1200).set_integer_casting())  
            # We optimize only the depth, so we fix all other parameters than the depth, using "partial".
            super().__init__(partial(self._ml_parametrization,
                                     noise_free=False, criterion="mse",
                                     min_samples_split=0.00001, dimension=dimension,
                                     regressor="decision_tree",
                                     alpha=1.0, learning_rate="no", 
                                     activation="no", solver="no"), parametrization)
            # For the evaluation, we remove the noise.
            self.evaluation_function = partial(self._ml_parametrization,  # type: ignore
                                               noise_free=True, criterion="mse", 
                                               dimension=dimension, min_samples_split=0.00001,
                                               regressor="decision_tree",        
                                               alpha=1.0, learning_rate="no", 
                                               activation="no", solver="no")
        elif regressor == "any":
            # First we define the list of parameters in the optimization
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),  # Depth, in case we use a decision tree.
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),  # Criterion for building the decision tree.
                min_samples_split=p.Log(lower=0.0000001, upper=1),  # Min ratio of samples in a node for splitting.
                regressor=p.Choice(["mlp", "decision_tree"]),  # Type of regressor.
                activation=p.Choice(["identity", "logistic", "tanh", "relu"]),  # Activation function, in case we use a net.
                solver=p.Choice(["lbfgs", "sgd", "adam"]),  # Numerical optimizer.
                learning_rate=p.Choice(["constant", "invscaling", "adaptive"]),  # Learning rate schedule.
                alpha=p.Log(lower=0.0000001, upper=1.),  # Complexity penalization.
            )        
            # Only the dimension is fixed, so "partial" is just used for fixing the dimension.
            # noise_free is False (meaning that we consider the cross-validation loss) during the optimization.
            super().__init__(partial(self._ml_parametrization, dimension=dimension, noise_free=False), parametrization)
            # For the evaluation we use the test set, which is big, so noise_free = True.
            self.evaluation_function = partial(self._ml_parametrization, dimension=dimension, noise_free=True)  # type: ignore
        elif regressor == "decision_tree":
            # We specify below the list of hyperparameters for the decision trees.
            parametrization = p.Instrumentation(
                depth=p.Scalar(lower=1, upper=1200).set_integer_casting(),
                criterion=p.Choice(["mse", "friedman_mse", "mae"]),
                min_samples_split=p.Log(lower=0.0000001, upper=1),
                regressor="decision_tree",
            )        
            # We use "partial" for fixing the parameters of the neural network, given that we work on the decision tree only.
            super().__init__(partial(self._ml_parametrization, dimension=dimension, noise_free=False,        
                                     alpha=1.0, learning_rate="no", regressor="decision_tree", 
                                     activation="no", solver="no"), parametrization)
            # For the test we just switch noise_free to True.
            self.evaluation_function = partial(self._ml_parametrization, dimension=dimension, criterion="mse",  # type: ignore
                                               min_samples_split=0.00001,
                                               regressor="decision_tree", noise_free=True,        
                                               alpha=1.0, learning_rate="no", 
                                               activation="no", solver="no")
        elif regressor == "mlp":
            # Let us define the parameters of the neural network.
            parametrization = p.Instrumentation(
                activation=p.Choice(["identity", "logistic", "tanh", "relu"]),
                solver=p.Choice(["lbfgs", "sgd", "adam"]),
                regressor="mlp",
                learning_rate=p.Choice(["constant", "invscaling", "adaptive"]),
                alpha=p.Log(lower=0.0000001, upper=1.),
            )        
            # And, using partial, we get rid of the parameters of the decision tree (we work on the neural net, not
            # on the decision tree).
            super().__init__(partial(self._ml_parametrization, dimension=dimension, noise_free=False,
                                     regressor="mlp", depth=-3, criterion="no", min_samples_split=0.1), parametrization)
            self.evaluation_function = partial(self._ml_parametrization, dimension=dimension,  # type: ignore
                                               regressor="mlp", noise_free=True, 
                                               depth=-3, criterion="no", min_samples_split=0.1)
        else:
            assert False, f"Problem type {regressor} undefined!"
        self.register_initialization(regressor=regressor, dimension=dimension)
