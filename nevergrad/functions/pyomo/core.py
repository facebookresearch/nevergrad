# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
from functools import partial
import types
import numpy as np
from nevergrad.parametrization import parameter as p
import pyomo.environ as pyo  # type: ignore
from pyomo.core.expr import current as pyo_expr  # type: ignore
#from . import photonics
from .. import base

def _make_pyomo_range_set_to_parametrization(pyo_model : pyo.Model, model_component : pyo.RangeSet, params : dict):
    # https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Sets.html
    # Refer to the implementation in pyomo/core/base/set.py
    if not((model_component.dimen == 1) and (len(model_component.ranges()) == 1)):
        raise NotImplementedError("RangeSet setting is currently not implemented")
    for i, r in enumerate(model_component.ranges()):
        params[model_component.name + '[' + str(i) + ']'] = p.Scalar(lower=r[0], upper=r[1])
    return params

def _make_pyomo_variable_to_parametrization(pyo_model : pyo.Model, model_component : pyo.Var, params : dict):
    # https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Sets.html
    # Refer to the implementation in pyomo/core/base/var.py
    if not model_component._data:
        return
    for k, v in model_component._data.items():
        params[model_component.name + '[' + str(k) + ']'] = p.Scalar(lower=v.bounds[0], upper=v.bounds[1])
    return params

class Pyomo(base.ExperimentFunction):
    """Function calling photonics code

    Parameters
    ----------
    name: str
        problem name
    model: pyomo.environ.model
        Pyomo model

    Returns
    -------
    float
        the fitness

    Notes
    -----
    - You will require an Pyomo installation (with pip: "pip install pyomo")
    - Any changes on the model externally can lead to unexpected behaviours.
    """

    def __init__(self, name : str, model : pyo.Model) -> None:
        if isinstance(model, pyo.ConcreteModel):
            self._model_instance = model
        else:
            self._model_instance = model.create_instance()
            raise NotImplementedError("AbstractModel is not supported yet.")

        instru_params: tp.Dict[Any, Any] = {}
        self.all_vars = []
        self.all_params = []
        self.all_constraints = []
        self.all_objectives = []

        #Relevant document: https://pyomo.readthedocs.io/en/stable/working_models.html
        for v in self._model_instance.component_objects(pyo.Var, active=True):
            self.all_vars.append(v)
            _make_pyomo_variable_to_parametrization(self._model_instance, v, instru_params)
        for v in self._model_instance.component_objects(pyo.Param, active=True):
            self.all_params.append(v)
        for v in self._model_instance.component_objects(pyo.Constraint, active=True):
            self.all_constraints.append(v)
        for v in self._model_instance.component_objects(pyo.Objective, active=True):
            self.all_objectives.append(v)

        if not self.all_objectives:
            raise NotImplementedError("Cannot find objective function")
        elif len(self.all_objectives) > 1:
            raise NotImplementedError("Multi-objective function is not supported yet.")

        if len(self.all_constraints) > 1:
            raise NotImplementedError("Constraint is not supported yet.")

        super().__init__(function=self._pyomo_obj_function_wrapper, parametrization=p.Instrumentation(**instru_params))
        self.register_initialization(**instru_params)
        self._descriptors.update(name=name)

    # pylint: disable=arguments-differ
    def evaluation_function(self, x: np.ndarray) -> float:  # type: ignore
        # pylint: disable=not-callable
        loss = self.function(x)
        base.update_leaderboard(f'{self.name},{self.parametrization.dimension}', loss, x, verbose=True)  # type: ignore
        return loss

    def _pyomo_obj_function_wrapper(self, **k_model_variables) -> float: # type: ignore
        for k, v in k_model_variables.items():
            exec(f"self._model_instance.{k} = {v}")
        return pyo.value(self.all_objectives[0]) #Single objective assumption
