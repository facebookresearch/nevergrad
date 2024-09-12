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
    archive: utils.Archive[utils.MultiValue],
    k: int,
    algorithm: str = "quad",
    degree: int = 2,
    shape: tp.Any = None,
    para: tp.Any = None,
) -> tp.ArrayLike:
    """Approximate optimum learnt from the k best.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
    """
    items = list(archive.items_as_arrays())
    dimension = len(items[0][0])
    if algorithm == "image":
        k = len(archive) // 6
    # Select the k best.
    first_k_individuals = sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("average"))[:k]
    if algorithm == "image":
        assert para is not None
        new_first_k_individuals = []
        for i in first_k_individuals:
            new_child = para.spawn_child()
            new_child.set_standardized_data(i[0])
            # print("- ", i[0][:3])
            # print(len(i[0]), len(new_child.value.flatten()))
            # print(i[0][:5], new_child.value.flatten()[:5])
            new_first_k_individuals += [new_child.value.flatten()]
    else:
        new_first_k_individuals = first_k_individuals
    # assert len(new_first_k_individuals[0]) == len(first_k_individuals[0][0])
    # first_k_individuals = in the representation space  (after [0])
    # new_first_k_individuals = in the space of real values for the user
    assert len(first_k_individuals) == k

    # Recenter the best.
    middle = np.array(sum(p[0] for p in first_k_individuals) / k)
    normalization = 1e-15 + np.sqrt(np.sum((first_k_individuals[-1][0] - first_k_individuals[0][0]) ** 2))
    if "image" == algorithm:
        middle = 0.0 * middle
        normalization = 1.0

    y = np.asarray([archive[c[0]].get_estimation("pessimistic") for c in first_k_individuals])
    if algorithm == "image":
        X = np.asarray([(c - middle) / normalization for c in new_first_k_individuals])
    else:
        X = np.asarray([(c[0] - middle) / normalization for c in new_first_k_individuals])
    # if algorithm == "image":
    #    print([(np.sum(x), np.min(x), np.max(x)) for x in X])
    from sklearn.preprocessing import PolynomialFeatures

    polynomial_features = PolynomialFeatures(degree=degree)

    def trans(X):
        if degree > 1:
            return polynomial_features.fit_transform(X)
        return X

    X2 = trans(X)
    if not max(y) - min(y) > 1e-20:  # better use "not" for dealing with nans
        raise MetaModelFailure
    y = (y - min(y)) / (max(y) - min(y))
    if algorithm == "neural":
        from sklearn.neural_network import MLPRegressor

        model = MLPRegressor(hidden_layer_sizes=(16, 16), solver="lbfgs")
    elif algorithm in ["image"]:
        from sklearn.svm import SVR
        import scipy.ndimage as ndimage

        # print(y)
        def rephrase(x):
            if shape is None:
                return x
            radii = [1 + int(0.3 * np.sqrt(shape[i])) for i in range(len(shape))]
            newx = ndimage.convolve(x.reshape(shape), np.ones(radii) / np.prod(radii))
            return newx

        def my_kernel(x, y):
            k = np.zeros(shape=(len(x), len(y)))
            for i in range(len(x)):
                for j in range(len(y)):
                    k[i][j] = np.exp(-500.0 * np.sum((rephrase(x[i]) - rephrase(y[j])) ** 2) / (len(x[0])))
            return k

        model = SVR(kernel=my_kernel, C=1e10, tol=1e-10)
        # print(X2.shape)
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
    success_rate = np.average(0.5 + 0.5 * np.sign(np.diff(ordered_model_outputs)))
    # if "image" == algorithm:
    # print([np.sum(x) for x in X2])
    # print("z", success_rate)  #, len(y), ordered_model_outputs)
    if not np.all(np.diff(ordered_model_outputs) > 0) and "image" != algorithm:
        raise MetaModelFailure("Unlearnable objective function.")
    if np.average(0.5 + 0.5 * np.sign(np.diff(ordered_model_outputs))) < 0.6:
        # if algorithm == "image":
        #    print("q")
        raise MetaModelFailure("Unlearnable objective function.")
    try:
        Powell = registry["Powell"]
        DE = registry["DE"]
        for cls in (Powell, DE):  # Powell excellent here, DE as a backup for thread safety.
            optimizer = cls(
                parametrization=para if (para is not None and algorithm == "image") else dimension,
                budget=45 * dimension + 30,
            )
            # limit to 20s at most
            optimizer.register_callback("ask", callbacks.EarlyStopping.timer(20))
            if "image" in algorithm:
                optimizer.suggest(new_first_k_individuals[0].reshape(shape))
                optimizer.suggest(new_first_k_individuals[1].reshape(shape))
                # print("k")
            try:
                minimum_point = optimizer.minimize(
                    lambda x: float(model.predict(trans(x.flatten()[None, :])))
                    # lambda x: float(model.predict(polynomial_features.fit_transform(x[None, :])))
                )
                minimum = minimum_point.value
            except RuntimeError:
                assert cls == Powell, "Only Powell is allowed to crash here."
            else:
                break
    except ValueError as e:
        # if "image" in algorithm:
        #    print("b", para, e)
        raise MetaModelFailure(f"Infinite meta-model optimum in learn_on_k_best: {e}.")
    if (
        float(model.predict(trans(minimum.flatten()[None, :]))) > y[len(y) // 3]
        and algorithm == "image"
        and success_rate < 0.9
    ):
        # print("bbb", float(model.predict(trans(minimum[None, :]))), y)
        raise MetaModelFailure("Not a good proposal.")
    if float(model.predict(trans(minimum[None, :]))) > y[0] and algorithm != "image":
        raise MetaModelFailure("Not a good proposal.")
    if algorithm == "image":
        # print(minimum)  # This is is the real space of the user.
        minimum = minimum_point.get_standardized_data(reference=para)
    # if float(model.predict(trans(minimum[None, :]))) > y[len(y) // 3]:
    #    if "image" in algorithm:
    #        print("bbb", float(model.predict(trans(minimum[None, :]))), "min:", y[0], "max:", y[-1], "avg:", np.average(y), "1/3:", y[len(y) // 3])
    #    raise MetaModelFailure("Not a good proposal.")
    if np.sum(minimum**2) > 1.0 and algorithm != "image":
        # if "image" in algorithm:
        #    print("d")
        raise MetaModelFailure("huge meta-model optimum in learn_on_k_best.")
    # if "image" in algorithm:
    # print("e" + "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    return middle + normalization * minimum.flatten()
