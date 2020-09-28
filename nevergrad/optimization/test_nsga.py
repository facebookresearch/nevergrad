# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=redefined-outer-name,unsubscriptable-object,unused-variable,unused-import
def test_doc_multiobjective() -> None:
    # DOC_MULTIOBJ_0
    import nevergrad as ng
    from nevergrad.functions import MultiobjectiveFunction
    import numpy as np

    f = MultiobjectiveFunction(multiobjective_function=lambda x: [np.sum(x**2), np.sum((x - 1)**2)], upper_bounds=[2.5, 2.5])
    print(f(np.array([1.0, 2.0])))

    optimizer = ng.optimizers.NSGAII(parametrization=3, budget=100)  # 3 is the dimension, 100 is the budget.
    recommendation = optimizer.minimize(f)

    # The function embeds its Pareto-front:
    print("My Pareto front:", [x[0][0] for x in f.pareto_front()])

    # It can also provide a subset:
    print("My Pareto front:", [x[0][0] for x in f.pareto_front(2, subset="random")])
    print("My Pareto front:", [x[0][0] for x in f.pareto_front(2, subset="loss-covering")])
    print("My Pareto front:", [x[0][0] for x in f.pareto_front(2, subset="domain-covering")])
    # DOC_MULTIOBJ_1
    assert len(f.pareto_front()) > 1
    assert len(f.pareto_front(2, "loss-covering")) == 2
    assert len(f.pareto_front(2, "domain-covering")) == 2
    assert len(f.pareto_front(2, "hypervolume")) == 2
    assert len(f.pareto_front(2, "random")) == 2

    # We can also run without upper_bounds: they are then computed automatically using "_auto_bound".
    f = MultiobjectiveFunction(multiobjective_function=lambda x: [np.sum(x**2), np.sum((x - 1)**2)])
    optimizer = ng.optimizers.NSGAII(parametrization=3, budget=100)  # 3 is the dimension, 100 is the budget.
    optimizer.minimize(f)
    assert len(f.pareto_front()) > 1

