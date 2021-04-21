# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import SkipTest
import numpy as np
import pytest
import nevergrad.common.typing as tp
import nevergrad as ng
from .. import base
from ..optimizerlib import registry
from . import core


def test_hypervolume_pareto_function() -> None:
    hvol = core.HypervolumePareto((100, 100))
    tuples = [
        (110, 110),  # -0 + distance
        (110, 90),  # -0 + distance
        (80, 80),  # -400 + distance
        (50, 50),  # -2500 + distance
        (50, 50),  # -2500 + distance
        (80, 80),  # -2500 + distance --> -2470
        (30, 60),  # [30,50]x[60,100] + [50,100]x[50,100] --> -2500 -800 = -3300
        (60, 30),
    ]  # [30,50]x[60,100] + [50,100]x[50,100] + [60,100]x[30,50] --> -2500 -800 -800= -4100
    values = []
    for tup in tuples:
        param = ng.p.Tuple(*(ng.p.Scalar(x) for x in tup))
        param._losses = np.array(tup)
        values.append(hvol.add(param))
    expected = [20, 10, -400, -2500.0, -2500.0, -2470.0, -3300.0, -4100.0]
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p.value for p in hvol.pareto_front()]
    expected_front = [(50, 50), (30, 60), (60, 30)]
    assert front == expected_front, f"Expected {expected_front} but got {front}"


def test_hypervolume_pareto_with_no_good_point() -> None:
    hvol = core.HypervolumePareto((100, 100))
    tuples = [(110, 110), (110, 90), (90, 110)]
    values = []
    for tup in tuples:
        param = ng.p.Tuple(*(ng.p.Scalar(x) for x in tup))
        param._losses = np.array(tup)
        values.append(hvol.add(param))
    expected = [20, 10, 10]
    assert values == expected, f"Expected {expected} but got {values}"
    front = [p.value for p in hvol.pareto_front()]
    expected_front = [(110, 90), (90, 110)]
    assert front == expected_front, f"Expected {expected_front} but got {front}"
    assert hvol._best_volume == -10  # used in xps


@pytest.mark.parametrize("losses", [(12, [3, 4]), ([3, 4], 12)])  # type: ignore
def test_num_losses_error(losses: tp.Tuple[tp.Any, tp.Any]) -> None:
    opt = ng.optimizers.CMA(parametrization=3, budget=100)
    cand = opt.ask()
    opt.tell(cand, losses[0])
    cand = opt.ask()
    with pytest.raises(ValueError):
        opt.tell(cand, losses[1])


def mofunc(array: np.ndarray) -> np.ndarray:
    return abs(array - 1)  # type: ignore


@pytest.mark.parametrize("name", registry)  # type: ignore
def test_optimizers_multiobjective(name: str) -> None:  # pylint: disable=redefined-outer-name
    if "BO" in name:
        raise SkipTest("BO is currently failing for unclear reasons")  # TODO solve
    with warnings.catch_warnings():
        # tests do not need to be efficient
        warnings.simplefilter("ignore", category=base.errors.InefficientSettingsWarning)
        optimizer = registry[name](parametrization=4, budget=100)
        candidate = optimizer.ask()
        optimizer.tell(candidate, mofunc(candidate.value))
    # to be tested
    optimizer.ask()
    to_be_tested = {"DE"}  # specific mo adapatation
    assert not to_be_tested - set(registry), "Testing some unknown optimizers"
    if optimizer.name in to_be_tested:
        optimizer.minimize(mofunc)


# pylint: disable=redefined-outer-name,unsubscriptable-object,
# pylint: disable=unused-variable,unused-import,reimported,import-outside-toplevel
def test_doc_multiobjective() -> None:
    # DOC_MULTIOBJ_OPT_0
    import nevergrad as ng
    import numpy as np

    def multiobjective(x):
        return [np.sum(x ** 2), np.sum((x - 1) ** 2)]

    print("Example: ", multiobjective(np.array([1.0, 2.0, 0])))
    # >>> Example: [5.0, 2.0]

    optimizer = ng.optimizers.CMA(parametrization=3, budget=100)

    # for all but DE optimizers, deriving a volume out of the losses,
    # it's not strictly necessary but highly advised to provide an
    # upper bound reference for the losses (if not provided, such upper
    # bound is automatically inferred with the first few "tell")
    optimizer.tell(ng.p.MultiobjectiveReference(), [5, 5])
    # note that you can provide a Parameter to MultiobjectiveReference,
    # which will be passed to the optimizer

    optimizer.minimize(multiobjective, verbosity=2)

    # The function embeds its Pareto-front:
    print("Pareto front:")
    for param in sorted(optimizer.pareto_front(), key=lambda p: p.losses[0]):
        print(f"{param} with losses {param.losses}")

    # >>> Array{(3,)}:[0. 0. 0.] with loss [0. 3.]
    #     Array{(3,)}:[0.39480968 0.98105712 0.55785803] with loss [1.42955333 0.56210368]
    #     Array{(3,)}:[1.09901515 0.97673712 0.97153943] with loss [3.10573857 0.01115516]

    # It can also provide subsets:
    print("Random subset:", optimizer.pareto_front(2, subset="random"))
    print("Loss-covering subset:", optimizer.pareto_front(2, subset="loss-covering"))
    print("Domain-covering subset:", optimizer.pareto_front(2, subset="domain-covering"))
    print("EPS subset:", optimizer.pareto_front(2, subset="EPS"))

    # DOC_MULTIOBJ_OPT_1
    assert len(optimizer.pareto_front()) > 1
    assert len(optimizer.pareto_front(2, "loss-covering")) == 2
    assert len(optimizer.pareto_front(2, "domain-covering")) == 2
    assert len(optimizer.pareto_front(2, "hypervolume")) == 2
    assert len(optimizer.pareto_front(2, "random")) == 2
    assert len(optimizer.pareto_front(2, "EPS")) == 2
