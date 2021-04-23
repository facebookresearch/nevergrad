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

from .ngautosklearn import get_parametrization, get_configuration, _eval_function, get_config_space
from .. import base


class AutoSKlearnBenchmark(base.ExperimentFunction):
    def __init__(
        self,
        openml_task_id: int,
        cv: int,
        time_budget_per_run: int,
        memory_limit: int,
        scoring_func: str = "balanced_accuracy",
        error_penalty: float = 1.0,
        overfitter: bool = False,
        random_state: tp.Optional[int] = None,
    ) -> None:

        self.openml_task_id = openml_task_id
        self.random_state = random_state
        self.cv = cv
        self.scoring_func = scoring_func
        self.memory_limit = memory_limit
        self.time_budget_per_run = time_budget_per_run
        self.error_penalty = error_penalty
        self.overfitter = overfitter
        self.evaluate_on_test = False
        self.eval_func = pynisher.enforce_limits(
            mem_in_mb=memory_limit, wall_time_in_s=self.time_budget_per_run
        )(_eval_function)
        openml_task = openml.tasks.get_task(openml_task_id)
        self.dataset_name = openml_task.get_dataset().name
        X, y = openml_task.get_X_and_y()
        split = openml_task.get_train_test_split_indices()
        self.X_train, self.y_train = X[split[0]], y[split[0]]
        self.X_test, self.y_test = X[split[1]], y[split[1]]

        self.config_space = get_config_space(
            X=self.X_train, y=self.y_train, time_budget_per_run=time_budget_per_run
        )
        parametrization = get_parametrization(self.config_space)
        parametrization = parametrization.set_name(f"time={time_budget_per_run}")

        self.add_descriptors(
            openml_task_id=openml_task_id,
            cv=cv,
            scoring_func=scoring_func,
            memory_limit=memory_limit,
            time_budget_per_run=time_budget_per_run,
            error_penalty=error_penalty,
            overfitter=overfitter,
            dataset_name=self.dataset_name,
        )
        self._descriptors.pop("random_state", None)  # remove it from automatically added descriptors
        self.best_loss = np.inf
        self.best_config = None
        super().__init__(self._simulate, parametrization)

    def _simulate(self, **x) -> float:
        if not self.evaluate_on_test:
            config = get_configuration(x, self.config_space)
            loss = self.eval_func(
                config=config,
                X=self.X_train,
                y=self.y_train,
                test_data=None,
                scoring_func=self.scoring_func,
                cv=self.cv,
                random_state=self.random_state,
            )
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_config = config
        else:
            loss = self.eval_func(
                config=self.best_config,
                X=self.X_train,
                y=self.y_train,
                test_data=(self.X_test, self.y_test),
                scoring_func=self.scoring_func,
                cv=self.cv,
                random_state=self.random_state,
            )

        return loss if isinstance(loss, float) else self.error_penalty

    def print_configuration(self, config):
        print(get_configuration(config.kwargs, self.config_space))

    def evaluation_function(self, *args) -> float:
        self.evaluate_on_test = not self.overfitter
        return super().evaluation_function(*args)
