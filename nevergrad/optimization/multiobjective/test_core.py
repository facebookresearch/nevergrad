# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad as ng
from . import core


def test_hypervolume_pareto_function() -> None:
    hvol = core.HypervolumePareto((100, 100))
    tuples = [(110, 110),     # -0 + distance
              (110, 90),      # -0 + distance
              (80, 80),       # -400 + distance
              (50, 50),       # -2500 + distance
              (50, 50),       # -2500 + distance
              (80, 80),       # -2500 + distance --> -2470
              (30, 60),       # [30,50]x[60,100] + [50,100]x[50,100] --> -2500 -800 = -3300
              (60, 30)]       # [30,50]x[60,100] + [50,100]x[50,100] + [60,100]x[30,50] --> -2500 -800 -800= -4100
    values = []
    for tup in tuples:
        param = ng.p.Tuple(*(ng.p.Scalar(x) for x in tup))
        param.loss = np.array(tup)
        values.append(hvol.add(param))
    expected = [10, 10, -400, -2500.0, -2500.0, -2470.0, -3300.0, -4100.0]
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p.value for p in hvol.pareto_front()]
    expected_front = [(50, 50), (30, 60), (60, 30)]
    assert front == expected_front, f"Expected {expected_front} but got {front}"


# pylint: disable=redefined-outer-name,unsubscriptable-object,unused-variable,unused-import,reimported,import-outside-toplevel
def test_doc_multiobjective() -> None:
    # DOC_MULTIOBJ_OPT_0
    import nevergrad as ng
    import numpy as np

    def multiobjective(x):
        return [np.sum(x**2), np.sum((x - 1)**2)]

    # upper_bounds=[2.5, 2.5])

    print("Example: ", multiobjective(np.array([1.0, 2.0, 0])))
    # >>> Example: [5.0, 2.0]

    # optimizer = ng.optimizers.CMA(parametrization=3, budget=100)  # 3 is the dimension, 100 is the budget.
    optimizer = ng.optimizers.OnePlusOne(parametrization=3, budget=100)  # 3 is the dimension, 100 is the budget.

    # it's not strictly necessary but highly advised to provide an upper bound reference for the losses
    # (if not provided, such upper bound is automatically computed with the first few "tell")
    optimizer.tell(ng.p.MultiobjectiveReference(), [5, 5])
    # note that you can provide a Parameter to MultiobjectiveReference, which will be passed to the
    # optimizer

    optimizer.minimize(multiobjective, verbosity=2)

    # The function embeds its Pareto-front:
    print("Pareto front:")
    for param in sorted(optimizer.pareto_front(), key=lambda p: p.loss[0]):  # type: ignore
        print(f"{param} with loss {param.loss}")

    # >>>[Array{(3,)}:[0. 0. 0.],
    #     Array{(3,)}:[ 0.74004948 -0.04942859  1.14188393],
    #     Array{(3,)}:[0.32063353 0.11387351 1.14188393]]

    # It can also provide a subset:
    print("Random subset:", optimizer.pareto_front(2, subset="random"))
    print("Loss-covering subset:", optimizer.pareto_front(2, subset="loss-covering"))
    print("Domain-covering subset:", optimizer.pareto_front(2, subset="domain-covering"))
    # DOC_MULTIOBJ_OPT_1
    assert len(optimizer.pareto_front()) > 1
    assert len(optimizer.pareto_front(2, "loss-covering")) == 2
    assert len(optimizer.pareto_front(2, "domain-covering")) == 2
    assert len(optimizer.pareto_front(2, "hypervolume")) == 2
    assert len(optimizer.pareto_front(2, "random")) == 2
    raise Exception
