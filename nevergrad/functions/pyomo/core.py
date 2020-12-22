# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
import numpy as np
import pyomo.environ as pyomo
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from .. import base


ParamDict = tp.Dict[str, p.Parameter]


def _convert_to_ng_name(pyomo_var_key: tp.Any) -> str:
    if isinstance(pyomo_var_key, str):
        return '"' + str(pyomo_var_key) + '"'
    else:
        return str(pyomo_var_key)


def _make_pyomo_range_set_to_parametrization(
    domain: pyomo.RangeSet, params: ParamDict, params_name: str
) -> ParamDict:
    # https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Sets.html
    # Refer to the implementation in pyomo/core/base/set.py
    ranges = list(domain.ranges())
    num_ranges = len(ranges)
    if num_ranges == 1 and (ranges[0].step in [-1, 0, 1]):
        if isinstance(ranges[0], pyomo.base.range.NumericRange):
            lb, ub = ranges[0].start, ranges[0].end
            if ranges[0].step < 0:
                lb, ub = ub, lb
            if (lb is not None) and (not ranges[0].closed[0]):
                lb = float(np.nextafter(lb, 1))
            if (ub is not None) and (not ranges[0].closed[1]):
                ub = float(np.nextafter(ub, -1))
            params[params_name] = p.Scalar(lower=lb, upper=ub)
            if ranges[0].step in [-1, 1]:
                # May consider using nested param
                params[params_name].set_integer_casting()  # type: ignore
        else:
            raise NotImplementedError(f"Cannot handle range type {type(ranges[0])}")
    elif isinstance(domain, pyomo.FiniteSimpleRangeSet):
        # Need to handle step size
        params[params_name] = p.Choice(
            [range(*r) for r in domain.ranges()]
        )  # Assume the ranges do not overlapped
    else:
        raise NotImplementedError(f"Cannot handle domain type {type(domain)}")
    return params


def _make_pyomo_variable_to_parametrization(model_component: pyomo.Var, params: ParamDict) -> ParamDict:
    # https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Sets.html
    # Refer to the implementation in pyomo/core/base/var.py
    # To further improve the readability function, we should find out how to represent {None: ng.p.Scalar(), 1: ng.p.Scalar()} in ng.p.Dict
    # We do not adopt nested parameterization, which will require type information between string and int.
    # Such conversion has to be done in _pyomo_obj_function_wrapper and _pyomo_constraint_wrapper, which slows down optimization.
    if not isinstance(model_component, (pyomo.base.var.IndexedVar, pyomo.base.var.SimpleVar)):
        raise NotImplementedError  # Normally, Pyomo will create a set for the indices used by a variable
    for k, v in model_component._data.items():
        if isinstance(v, pyomo.base.var._GeneralVarData):
            if v.is_fixed():
                raise NotImplementedError
            if k is None:
                params_name = str(model_component.name)
            else:
                params_name = f"{model_component.name}[{_convert_to_ng_name(k)}]"
            if isinstance(v.domain, pyomo.RangeSet):
                params = _make_pyomo_range_set_to_parametrization(v.domain, params, params_name)
            elif isinstance(v.domain, pyomo.Set) and v.domain.isfinite():
                if v.domain.isordered():
                    params[params_name] = p.Choice(list(v.domain.ordered_data()))
                else:
                    params[params_name] = p.Choice(list(v.domain.data()))
            else:
                raise NotImplementedError(f"Cannot handle domain type {type(v.domain)}")
        else:
            raise NotImplementedError(f"Cannot handle variable type {type(v)}")
    return params


class Pyomo(base.ExperimentFunction):
    """Function calling Pyomo model

    Parameters
    ----------
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

    def __init__(self, model: pyomo.Model) -> None:
        if isinstance(model, pyomo.ConcreteModel):
            self._model_instance = model.clone()  # To enable the objective function to run in parallel
        else:
            raise NotImplementedError(
                "AbstractModel is not supported. Please use create_instance() in Pyomo to create a model instance."
            )

        instru_params: ParamDict = {}
        self.all_vars: tp.List[pyomo.Var] = []
        self.all_params: tp.List[pyomo.Param] = []
        self.all_constraints: tp.List[pyomo.Constraint] = []
        self.all_objectives: tp.List[pyomo.Objective] = []

        # Relevant document: https://pyomo.readthedocs.io/en/stable/working_models.html

        for v in self._model_instance.component_objects(pyomo.Var, active=True):
            self.all_vars.append(v)
            _make_pyomo_variable_to_parametrization(v, instru_params)
        for v in self._model_instance.component_objects(pyomo.Param, active=True):
            self.all_params.append(v)
        for v in self._model_instance.component_objects(pyomo.Constraint, active=True):
            self.all_constraints.append(v)
        for v in self._model_instance.component_objects(pyomo.Objective, active=True):
            # if v.sense == -1:
            #    print(f"Only minimization problem is supported. The value of the objective function {v.name} will be multiplied by -1.")
            self.all_objectives.append(v)

        if not self.all_objectives:
            raise NotImplementedError("Cannot find objective function")

        if len(self.all_objectives) > 1:
            raise NotImplementedError("Multi-objective function is not supported yet.")

        self._value_assignment_code_obj = ""

        instru = p.Instrumentation(**instru_params).set_name("")
        for c_idx in range(0, len(self.all_constraints)):
            instru.register_cheap_constraint(partial(self._pyomo_constraint_wrapper, c_idx))
        super().__init__(
            function=partial(self._pyomo_obj_function_wrapper, 0), parametrization=instru
        )  # Single objective

        exp_tag = ",".join([n.name for n in self.all_objectives])
        exp_tag += "|" + ",".join([n.name for n in self.all_vars])
        exp_tag += "|" + ",".join([n.name for n in self.all_constraints])
        self.add_descriptors(name=exp_tag)

    def _pyomo_value_assignment(self, k_model_variables: tp.Dict[str, tp.Any]) -> None:
        if self._value_assignment_code_obj == "":
            code_str = ""
            for k in k_model_variables:
                code_str += f"self._model_instance.{k} = k_model_variables['{k}']\n"
            self._value_assignment_code_obj = compile(code_str, "<string>", "exec")
        # TODO find a way to avoid exec
        exec(self._value_assignment_code_obj)  # pylint: disable=exec-used

    def _pyomo_obj_function_wrapper(self, i: int, **k_model_variables: tp.Dict[str, tp.Any]) -> float:
        self._pyomo_value_assignment(k_model_variables)
        return float(
            pyomo.value(self.all_objectives[i] * self.all_objectives[i].sense)
        )  # Single objective assumption

    def _pyomo_constraint_wrapper(self, i: int, instru: tp.ArgsKwargs) -> bool:
        k_model_variables = instru[1]
        # Combine all constraints into single one
        self._pyomo_value_assignment(k_model_variables)
        if isinstance(self.all_constraints[i], pyomo.base.constraint.SimpleConstraint):
            return bool(pyomo.value(self.all_constraints[i].expr(self._model_instance)))
        elif isinstance(self.all_constraints[i], pyomo.base.constraint.IndexedConstraint):
            ret = True
            for _, c in self.all_constraints[i].items():
                ret = ret and pyomo.value(c.expr(self._model_instance))
                if not ret:
                    break
            return ret
        else:
            raise NotImplementedError(
                f"Constraint type {self.all_constraints[i].ctype} is not supported yet."
            )
