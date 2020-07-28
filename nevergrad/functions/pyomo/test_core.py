# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import division
import numpy as np
import pyomo.environ as pyo  # type: ignore
import nevergrad as ng
from . import core

def test_concrete_model_without_constraints():
    pyo_model = pyo.ConcreteModel()
    pyo_model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    pyo_model.OBJ = pyo.Objective(expr = (pyo_model.x[1]-0.5)**2 + (pyo_model.x[2]-0.5)**2)

    func = core.Pyomo('Pyomo Model', pyo_model)
    optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)

    np.testing.assert_almost_equal(recommendation.kwargs['x[1]'], 0.5, decimal=1)
    np.testing.assert_almost_equal(recommendation.kwargs['x[2]'], 0.5, decimal=1) 

def test_concrete_model_with_constraints():
    pyo_model = pyo.ConcreteModel()
    pyo_model.x = pyo.Var([1,2], domain=pyo.NonNegativeReals)
    pyo_model.OBJ = pyo.Objective(expr = 2*pyo_model.x[1] + 3*pyo_model.x[2] + 5)
    #pyo_model.Constraint1 = pyo.Constraint(expr = 3*pyo_model.x[1] + 4*pyo_model.x[2] >= 1)

    func = core.Pyomo('Pyomo Model', pyo_model)
    optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)

    print(recommendation)
