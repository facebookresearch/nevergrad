# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Trained policies were extracted from https://github.com/modestyachts/ARS
# under their own license. See ARS_LICENSE file in this file's directory

import gym
import openml
import pynisher
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from .. import base
from ConfigSpace.read_and_write import json
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from .ngautosklearn import get_parametrization, get_configuration
from autosklearn.pipeline.classification import SimpleClassificationPipeline
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import traceback

def _eval_function(config, X, y, scoring_func, cv, random_state):
    try:
        classifier = SimpleClassificationPipeline(config=config, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(classifier,
                                     X,
                                     y,
                                     cv=StratifiedKFold(n_splits=cv, random_state=random_state, shuffle=True),
                                     scoring=scoring_func,
                                     n_jobs=1,
                                     )
        return 1 - np.mean(scores)
    except Exception as e:
        traceback.print_tb("WWWWW", e.__traceback__)
        return 1



class AutoSKlearnBenchmark(base.ExperimentFunction):

    def __init__(self, openml_task_id: int, cv:int, time_budget_per_run: int,
                 memory_limit: int, scoring_func: str = "balanced_accuracy",
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

        self.eval_func = pynisher.enforce_limits(mem_in_mb=memory_limit,
                                                 wall_time_in_s=self.time_budget_per_run)(_eval_function)
        openml_task = openml.tasks.get_task(openml_task_id)
        self.dataset_name = openml_task.get_dataset().name
        self.X, self.y = openml_task.get_X_and_y()

        self.add_descriptors(openml_task_id=openml_task_id, cv=cv,
                             scoring_func=scoring_func,
                             memory_limit=memory_limit,
                             time_budget_per_run=time_budget_per_run,
                             dataset_name=self.dataset_name)
        self._descriptors.pop("random_state", None)  # remove it from automatically added descriptors

    def _simulate(self, **x) -> float:
        config = get_configuration(x, self.config_space)

        loss = self.eval_func(config=config, X=self.X, y=self.y,
                              scoring_func=self.scoring_func,
                              cv=self.cv, random_state=self.random_state)
        return loss if loss is not None else 1

    # def evaluation_function(self, *recommendations: p.Parameter) -> float:
    #     assert len(recommendations) == 1, "Should not be a pareto set for a monoobjective function"
    #     x = recommendations[0].value
    #     # pylint: disable=not-callable
    #     loss = self.function(x)
    #     assert isinstance(loss, float)
    #     base.update_leaderboard(f"{self.env_name},{self.parametrization.dimension}", loss, x, verbose=True)
    #     return loss
