# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
import os
import numpy as np
import pyomo.environ as pyomo
import nevergrad as ng
from . import core


def test_concrete_model_without_constraints() -> None:
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var([1, 2], domain=pyomo.NonNegativeReals)
    model.obj = pyomo.Objective(expr=(model.x[1] - 0.5) ** 2 + (model.x[2] - 0.5) ** 2)

    func = core.Pyomo(model)
    optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)

    np.testing.assert_almost_equal(recommendation.kwargs["x[1]"], 0.5, decimal=1)
    np.testing.assert_almost_equal(recommendation.kwargs["x[2]"], 0.5, decimal=1)


def square(m: tp.Any) -> float:
    return pyomo.quicksum((m.x[i] - 0.5) ** 2 for i in m.x)


def test_concrete_model_with_constraints() -> None:
    model = pyomo.ConcreteModel()
    model.x = pyomo.Var([0, 1], domain=pyomo.Reals)
    model.obj = pyomo.Objective(rule=square)
    model.Constraint1 = pyomo.Constraint(rule=lambda m: m.x[0] >= 1)
    model.Constraint2 = pyomo.Constraint(rule=lambda m: m.x[1] >= 0.8)

    func = core.Pyomo(model)
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)

    np.testing.assert_almost_equal(recommendation.kwargs["x[0]"], 1.0, decimal=1)
    np.testing.assert_almost_equal(recommendation.kwargs["x[1]"], 0.8, decimal=1)


def test_abstract_model_with_constraints() -> None:
    abs_model = pyomo.AbstractModel()
    abs_model.F = pyomo.Set()
    abs_model.Xmin = pyomo.Param(abs_model.F, within=pyomo.Reals, default=0.0)
    abs_model.x = pyomo.Var(abs_model.F, within=pyomo.Reals)
    abs_model.constraints = pyomo.Constraint(abs_model.F, rule=lambda m, i: m.x[i] >= m.Xmin[i])
    abs_model.obj = pyomo.Objective(rule=square)

    # Load the values of the parameters from external file
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "test_model_1.dat")
    model = abs_model.create_instance(filename)

    func = core.Pyomo(model)
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=200)
    recommendation = optimizer.minimize(func.function)

    np.testing.assert_almost_equal(recommendation.kwargs['x["New York"]'], model.Xmin["New York"], decimal=1)
    np.testing.assert_almost_equal(
        recommendation.kwargs['x["Hong Kong"]'], model.Xmin["Hong Kong"], decimal=1
    )


def test_pyomo_set() -> None:
    def square2(m: tp.Any) -> float:
        return (m.x - 1) ** 2  # type: ignore

    model = pyomo.ConcreteModel()
    model.P = pyomo.Set(initialize=list(range(1, 11)))
    model.Q = pyomo.Set(initialize=list(range(6, 16)))
    model.R = model.P ^ model.Q  # XOR
    model.x = pyomo.Var(domain=model.R)
    model.obj = pyomo.Objective(rule=square2)
    model.constraint1 = pyomo.Constraint(rule=lambda m: m.x >= 2)

    func = core.Pyomo(model)
    func.parametrization.random_state.seed(12)
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)

    np.testing.assert_almost_equal(recommendation.kwargs["x"], 2.0, decimal=1)
