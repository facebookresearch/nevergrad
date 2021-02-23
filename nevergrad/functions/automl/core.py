# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Trained policies were extracted from https://github.com/modestyachts/ARS
# under their own license. See ARS_LICENSE file in this file's directory

import nevergrad.common.typing as tp
import numpy as np
import openml
import pynisher
from ConfigSpace.read_and_write import json
from autosklearn.pipeline.classification import SimpleClassificationPipeline
from nevergrad.parametrization import parameter as p
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from .ngautosklearn import get_parametrization, get_configuration
from .. import base


def warn(*args, **kwargs):
    pass


import warnings
warnings.warn = warn


def _eval_function(config, X, y, scoring_func, cv, random_state, test_data=None):
    try:
        classifier = SimpleClassificationPipeline(config=config, random_state=random_state)
        scorer = get_scorer(scoring_func)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if test_data is None:
                scores = cross_val_score(classifier,
                                         X,
                                         y,
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


class AutoSKlearnBenchmark(base.ExperimentFunction):

    def __init__(self, openml_task_id: int, cv: int, time_budget_per_run: int,
                 memory_limit: int, scoring_func: str = "balanced_accuracy",
                 error_penalty: int = 1,
                 random_state: tp.Optional[int] = None) -> None:

        parametrization, self.config_space = get_parametrization()
        parametrization = parametrization.set_name(f"time={time_budget_per_run}")
        super().__init__(self._simulate,
                         parametrization)
        self.openml_task_id = openml_task_id
        self.random_state = random_state
        self.cv = cv
        self.scoring_func = scoring_func
        self.memory_limit = memory_limit
        self.time_budget_per_run = time_budget_per_run
        self.error_penalty = error_penalty

        self.eval_func = pynisher.enforce_limits(mem_in_mb=memory_limit,
                                                 wall_time_in_s=self.time_budget_per_run)(_eval_function)
        openml_task = openml.tasks.get_task(openml_task_id)
        self.dataset_name = openml_task.get_dataset().name
        X, y = openml_task.get_X_and_y()
        split = openml_task.get_train_test_split_indices()
        self.X_train, self.y_train = X[split[0]], y[split[0]]
        self.X_test, self.y_test = X[split[1]], y[split[1]]
        self.evaluate_on_test = False

        self.add_descriptors(openml_task_id=openml_task_id, cv=cv,
                             scoring_func=scoring_func,
                             memory_limit=memory_limit,
                             time_budget_per_run=time_budget_per_run,
                             error_penalty=error_penalty,
                             dataset_name=self.dataset_name)
        self._descriptors.pop("random_state", None)  # remove it from automatically added descriptors

    def _simulate(self, **x) -> float:
        config = get_configuration(x, self.config_space)

        if not self.evaluate_on_test:
            loss = self.eval_func(config=config, X=self.X_train, y=self.y_train,
                                  test_data=None,
                                  scoring_func=self.scoring_func,
                                  cv=self.cv, random_state=self.random_state)
        else:
            loss = self.eval_func(config=config, X=self.X_train, y=self.y_train,
                                  test_data=(self.X_test, self.y_test),
                                  scoring_func=self.scoring_func,
                                  cv=self.cv, random_state=self.random_state)

        return loss if isinstance(loss, float) else self.error_penalty
