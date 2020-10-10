# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp
import os
import numpy as np
import pyomo.environ as pyomo
import nevergrad as ng
from . import solver


def test_solver_cambodian() -> None:
    solver.download_dataset("cambodian")
    dataset_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dataset_path, "pownet_cambodian", "Model_withdata", 'pownet_data_camb_2016.dat')
    solver.pownet_solver(dataset_path, model_type="cambodian")

    # model = pyomo.ConcreteModel()
    # model.x = pyomo.Var([1, 2], domain=pyomo.NonNegativeReals)
    # model.obj = pyomo.Objective(expr=(model.x[1] - 0.5)**2 + (model.x[2] - 0.5)**2)

    # func = core.Pyomo(model)
    # optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=100)
    # recommendation = optimizer.minimize(func.function)

    # np.testing.assert_almost_equal(recommendation.kwargs['x[1]'], 0.5, decimal=1)
    # np.testing.assert_almost_equal(recommendation.kwargs['x[2]'], 0.5, decimal=1)

def test_solver_tiny() -> None:
    dataset_path = os.path.dirname(os.path.realpath(__file__))
    dataset_path = os.path.join(dataset_path, "pownet_tiny", "model_data", 'pownet_data_tiny.dat')
    solver.pownet_solver(dataset_path, model_type="tiny")
