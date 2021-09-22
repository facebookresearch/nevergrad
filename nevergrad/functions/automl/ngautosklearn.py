# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import ConfigSpace as cs  # type: ignore
import nevergrad as ng
import numpy as np
import scipy
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

try:
    from autosklearn.constants import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION  # type: ignore
    from autosklearn.util.pipeline import get_configuration_space  # type: ignore
    from autosklearn.pipeline.classification import SimpleClassificationPipeline  # type: ignore
except ImportError:
    raise ImportError("Auto-Sklearn not installed. Run: python -m pip install auto-sklearn==0.11.0")


def _eval_function(
    config: cs.Configuration, X, y, scoring_func: str, cv: int, random_state: int, test_data: tuple
):
    try:
        classifier = SimpleClassificationPipeline(config=config, random_state=random_state)
        scorer = get_scorer(scoring_func)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if test_data is None:
                scores = cross_val_score(
                    estimator=classifier,
                    X=X,
                    y=y,
                    cv=StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True),
                    scoring=scorer,
                    n_jobs=1,
                )
                return 1 - np.mean(scores)
            else:
                classifier.fit(X, y)
                return 1 - scorer(classifier, test_data[0], test_data[1])
    except Exception:
        return 1


def check_configuration(config_space, values):
    val_dict = to_dict(values[1])
    try:
        cs.Configuration(configuration_space=config_space, values=val_dict, allow_inactive_with_values=False)
    except Exception:
        return False
    return True


def get_config_space(X, y):
    dataset_properties = {
        "task": BINARY_CLASSIFICATION if len(np.unique(y)) == 2 else MULTICLASS_CLASSIFICATION,
        "is_sparse": scipy.sparse.issparse(X),
    }
    return get_configuration_space(dataset_properties)


def get_instrumention(param):
    if isinstance(param, cs.hyperparameters.CategoricalHyperparameter):
        return ng.p.Choice(param.choices)
    elif isinstance(param, cs.hyperparameters.UniformIntegerHyperparameter):
        if param.log == False:
            return ng.p.Scalar(
                lower=param.lower, upper=param.upper, init=param.default_value
            ).set_integer_casting()
        else:
            return ng.p.Log(
                lower=param.lower, upper=param.upper, init=param.default_value
            ).set_integer_casting()
    elif isinstance(param, cs.hyperparameters.UniformFloatHyperparameter):
        if param.log == False:
            return ng.p.Scalar(lower=param.lower, upper=param.upper, init=param.default_value)
        else:
            return ng.p.Log(lower=param.lower, upper=param.upper, init=param.default_value)
    elif isinstance(param, cs.hyperparameters.Constant):
        return param.value
    raise Exception(r"{param} type not known")


def get_parametrization(config_space: cs.ConfigurationSpace):
    base_pipeline = [
        "balancing:strategy",
        "classifier:__choice__",
        "data_preprocessor:feature_type:categorical_transformer:categorical_encoding:__choice__",
        "data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__",
        "data_preprocessor:feature_type:numerical_transformer:imputation:strategy",
        "data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__",
        "feature_preprocessor:__choice__",
        "data_preprocessor:__choice__",
    ]

    params = {}

    for param in config_space.get_hyperparameters():
        if param.name in base_pipeline:
            if param.name in [
                "classifier:__choice__",
                "feature_preprocessor:__choice__",
                "data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__",
                "data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__",
            ]:
                params[param.name] = ng.p.Choice(
                    [
                        ng.p.Tuple(
                            param_choice,
                            ng.p.Dict(
                                **{
                                    hp.name: get_instrumention(hp)
                                    for hp in config_space.get_hyperparameters()
                                    if param_choice in hp.name
                                }
                            ),
                        )
                        for param_choice in param.choices
                    ]
                )
            else:
                params[param.name] = get_instrumention(param)

    inst = ng.p.Instrumentation(**params)
    from functools import partial

    constraint_check_func = partial(check_configuration, config_space)
    inst.register_cheap_constraint(constraint_check_func)
    return inst


def get_configuration(values, config_space):
    val_dict = to_dict(values)
    return cs.Configuration(
        configuration_space=config_space, values=val_dict, allow_inactive_with_values=True
    )


def to_dict(values):
    clf = values["classifier:__choice__"]
    features = values["feature_preprocessor:__choice__"]
    data_preprocessor = values["data_preprocessor:__choice__"]
    trans_cat = values[
        "data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__"
    ]
    trans_num = values["data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__"]
    del values["classifier:__choice__"]
    del values["feature_preprocessor:__choice__"]
    del values["data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__"]
    del values["data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__"]
    values["classifier:__choice__"] = clf[0]
    values.update(clf[1])
    values["feature_preprocessor:__choice__"] = features[0]
    values.update(features[1])
    values["data_preprocessor:__choice__"] = data_preprocessor
    values[
        "data_preprocessor:feature_type:categorical_transformer:category_coalescence:__choice__"
    ] = trans_cat[0]
    if len(trans_cat[1]) > 0:
        values.update(trans_cat[1])
    values["data_preprocessor:feature_type:numerical_transformer:rescaling:__choice__"] = trans_num[0]
    if len(trans_num[1]) > 0:
        values.update(trans_num[1])
    return values
