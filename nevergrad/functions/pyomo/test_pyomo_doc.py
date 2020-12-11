# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import numpy as np

# pylint: disable=reimported,redefined-outer-name,unused-variable,unsubscriptable-object, unused-argument
# pylint: disable=import-outside-toplevel


def test_concrete_model_example() -> None:
    # DOC_CONCRETE_0
    import pyomo.environ as pyomo

    def square(m):
        return pyomo.quicksum((m.x[i] - 0.5) ** 2 for i in m.x)

    model = pyomo.ConcreteModel()
    model.x = pyomo.Var([0, 1], domain=pyomo.Reals)
    model.obj = pyomo.Objective(rule=square)
    model.Constraint1 = pyomo.Constraint(rule=lambda m: m.x[0] >= 1)
    model.Constraint2 = pyomo.Constraint(rule=lambda m: m.x[1] >= 0.8)
    # DOC_CONCRETE_1

    # DOC_CONCRETE_10
    import nevergrad as ng
    import nevergrad.functions.pyomo as ng_pyomo

    # DOC_CONCRETE_11

    # DOC_CONCRETE_100
    func = ng_pyomo.Pyomo(model)
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)
    # DOC_CONCRETE_101

    # DOC_CONCRETE_1000
    print(recommendation.kwargs["x[0]"])
    print(recommendation.kwargs["x[1]"])
    # DOC_CONCRETE_1001

    np.testing.assert_almost_equal(recommendation.kwargs["x[0]"], 1.0, decimal=1)
    np.testing.assert_almost_equal(recommendation.kwargs["x[1]"], 0.8, decimal=1)


def test_abstract_model_example() -> None:
    import pyomo.environ as pyomo

    def square(m):
        return pyomo.quicksum((m.x[i] - 0.5) ** 2 for i in m.x)

    abstract_model = pyomo.AbstractModel()
    abstract_model.F = pyomo.Set()
    abstract_model.Xmin = pyomo.Param(abstract_model.F, within=pyomo.Reals, default=0.0)
    abstract_model.x = pyomo.Var(abstract_model.F, within=pyomo.Reals)
    abstract_model.constraints = pyomo.Constraint(abstract_model.F, rule=lambda m, i: m.x[i] >= m.Xmin[i])
    abstract_model.obj = pyomo.Objective(rule=square)

    import nevergrad as ng
    import nevergrad.functions.pyomo as ng_pyomo

    # Load the values of the parameters from external file
    data_path = str(Path(__file__).with_name("test_model_1.dat"))

    # DOC_ABSTRACT_100
    data = pyomo.DataPortal()
    data.load(filename=data_path, model=abstract_model)
    model = abstract_model.create_instance(data)
    # DOC_ABSTRACT_101

    func = ng_pyomo.Pyomo(model)
    optimizer = ng.optimizers.OnePlusOne(parametrization=func.parametrization, budget=200)
    recommendation = optimizer.minimize(func.function)

    np.testing.assert_almost_equal(recommendation.kwargs['x["New York"]'], model.Xmin["New York"], decimal=1)
    np.testing.assert_almost_equal(
        recommendation.kwargs['x["Hong Kong"]'], model.Xmin["Hong Kong"], decimal=1
    )
