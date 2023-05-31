# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad.common.typing as tp
from . import utils
from .base import registry
from . import callbacks


class MetaModelFailure(ValueError):
    """Sometimes the optimum of the metamodel is at infinity."""


def learn_on_k_best(
    archive: utils.Archive[utils.MultiValue], k: int, algorithm: str = "quad"
) -> tp.ArrayLike:
    """Approximate optimum learnt from the k best.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
    """
    items = list(archive.items_as_arrays())
    dimension = len(items[0][0])

    # Select the k best.
    first_k_individuals = sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("average"))[:k]
    assert len(first_k_individuals) == k

    # Recenter the best.
    middle = np.array(sum(p[0] for p in first_k_individuals) / k)
    normalization = 1e-15 + np.sqrt(np.sum((first_k_individuals[-1][0] - first_k_individuals[0][0]) ** 2))
    y = np.asarray([archive[c[0]].get_estimation("pessimistic") for c in first_k_individuals])
    X = np.asarray([(c[0] - middle) / normalization for c in first_k_individuals])

    from sklearn.preprocessing import PolynomialFeatures

    polynomial_features = PolynomialFeatures(degree=2)
    X2 = polynomial_features.fit_transform(X)
    if not max(y) - min(y) > 1e-20:  # better use "not" for dealing with nans
        raise MetaModelFailure
    y = (y - min(y)) / (max(y) - min(y))
    if algorithm == "neural":
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(hidden_layer_sizes=(16, 16), solver="lbfgs")
    elif algorithm in ["svm", "svr"]:
        from sklearn.svm import SVR

        model = SVR()
    elif algorithm == "rf":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor()
    else:
        assert algorithm == "quad", f"Metamodelling algorithm {algorithm} not recognized."
        # We need SKLearn.
        from sklearn.linear_model import LinearRegression

        # Fit a linear model.
        model = LinearRegression()

    model.fit(X2, y)
    # Check model quality.
    model_outputs = model.predict(X2)
    indices = np.argsort(y)
    ordered_model_outputs = [model_outputs[i] for i in indices]
    if not np.all(np.diff(ordered_model_outputs) > 0):
        raise MetaModelFailure("Unlearnable objective function.")

    try:
        Powell = registry["Powell"]
        DE = registry["DE"]
        for cls in (Powell, DE):  # Powell excellent here, DE as a backup for thread safety.
            optimizer = cls(parametrization=dimension, budget=45 * dimension + 30)
            # limit to 20s at most
            optimizer.register_callback("ask", callbacks.EarlyStopping.timer(20))
            try:
                minimum = optimizer.minimize(
                    lambda x: float(model.predict(polynomial_features.fit_transform(x[None, :])))
                ).value
            except RuntimeError:
                assert cls == Powell, "Only Powell is allowed to crash here."
            else:
                break
    except ValueError:
        raise MetaModelFailure("Infinite meta-model optimum in learn_on_k_best.")
    if float(model.predict(polynomial_features.fit_transform(minimum[None, :]))) > y[0]:
        raise MetaModelFailure("Not a good proposal.")
    if np.sum(minimum**2) > 1.0:
        raise MetaModelFailure("huge meta-model optimum in learn_on_k_best.")
    return middle + normalization * minimum
