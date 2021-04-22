# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
import nevergrad as ng
from nevergrad.common.tools import flatten
from .optimizerlib import registry
from .externalbo import _hp_parametrization_to_dict, _hp_dict_to_parametrization


@pytest.mark.parametrize(  # type: ignore
    "parametrization,has_transform",
    [
        (ng.p.Choice(list(range(10))), True),
        (ng.p.Scalar(lower=0, upper=1), True),
        (ng.p.Scalar(lower=0, upper=10).set_integer_casting(), True),
        (ng.p.Log(lower=1e-3, upper=1e3), True),
        (ng.p.Array(init=np.zeros(10)), True),
        (ng.p.Instrumentation(ng.p.Scalar(lower=0, upper=1), a=ng.p.Choice(list(range(10)))), False),
        (
            ng.p.Instrumentation(
                a=ng.p.Choice([ng.p.Scalar(lower=0, upper=1), ng.p.Scalar(lower=100, upper=1000)])
            ),
            True,
        ),
        (
            ng.p.Instrumentation(
                a=ng.p.Choice(
                    [
                        ng.p.Choice(list(range(10))),
                        ng.p.Scalar(lower=0, upper=1),
                    ]
                )
            ),
            False,
        ),
        (
            ng.p.Instrumentation(
                a=ng.p.Choice(
                    [
                        ng.p.Instrumentation(
                            b=ng.p.Choice(list(range(10))), c=ng.p.Log(lower=1e-3, upper=1e3)
                        ),
                        ng.p.Instrumentation(
                            d=ng.p.Scalar(lower=0, upper=1), e=ng.p.Log(lower=1e-3, upper=1e3)
                        ),
                    ]
                )
            ),
            False,
        ),
    ],
)
def test_hyperopt(parametrization, has_transform) -> None:
    optim1 = registry["HyperOpt"](parametrization=parametrization, budget=5)
    optim2 = registry["HyperOpt"](parametrization=parametrization.copy(), budget=5)
    for it in range(4):
        cand = optim1.ask()
        optim1.tell(cand, 0)  # Tell asked
        del cand._meta["trial_id"]
        optim2.tell(cand, 0)  # Tell not asked
        assert flatten(optim1.trials._dynamic_trials[it]["misc"]["vals"]) == pytest.approx(  # type: ignore
            flatten(optim2.trials._dynamic_trials[it]["misc"]["vals"])  # type: ignore
        )

    assert optim1.trials.new_trial_ids(1) == optim2.trials.new_trial_ids(1)  # type: ignore
    assert optim1.trials.new_trial_ids(1)[0] == (it + 2)  # type: ignore
    assert (optim1._transform is not None) == has_transform  # type: ignore

    # Test parallelization
    opt = registry["HyperOpt"](parametrization=parametrization, budget=30, num_workers=5)
    for k in range(40):
        cand = opt.ask()
        if not k:
            opt.tell(cand, 1)


@pytest.mark.parametrize(  # type: ignore
    "parametrization,values",
    [
        (
            ng.p.Instrumentation(
                a=ng.p.Choice([ng.p.Choice(list(range(10))), ng.p.Scalar(lower=0, upper=1)])
            ),
            [
                (((), {"a": 0.5}), {"a": [1], "a__1": [0.5]}, {"args": {}, "kwargs": {"a": 0.5}}),
                (((), {"a": 1}), {"a": [0], "a__0": [1]}, {"args": {}, "kwargs": {"a": 1}}),
            ],
        ),
        (
            ng.p.Instrumentation(ng.p.Scalar(lower=0, upper=1), a=ng.p.Choice(list(range(10)))),
            [
                (((0.5,), {"a": 3}), {"0": [0.5], "a": [3]}, {"args": {"0": 0.5}, "kwargs": {"a": 3}}),
                (((0.99,), {"a": 0}), {"0": [0.99], "a": [0]}, {"args": {"0": 0.99}, "kwargs": {"a": 0}}),
            ],
        ),
        (
            ng.p.Instrumentation(
                a=ng.p.Choice(
                    [
                        ng.p.Instrumentation(
                            b=ng.p.Choice(list(range(10))), c=ng.p.Log(lower=1e-3, upper=1e3)
                        ),
                        ng.p.Instrumentation(
                            d=ng.p.Scalar(lower=0, upper=1), e=ng.p.Log(lower=1e-3, upper=1e3)
                        ),
                    ]
                )
            ),
            [
                (
                    ((), {"a": ((), {"d": 0.5, "e": 1.0})}),
                    {"a": [1], "d": [0.5], "e": [1.0]},
                    {"args": {}, "kwargs": {"a": {"args": {}, "kwargs": {"d": 0.5, "e": 1.0}}}},
                ),
                (
                    ((), {"a": ((), {"b": 0, "c": 0.014})}),
                    {"a": [0], "b": [0], "c": [0.014]},
                    {"args": {}, "kwargs": {"a": {"args": {}, "kwargs": {"b": 0, "c": 0.014}}}},
                ),
            ],
        ),
    ],
)
def test_hyperopt_helpers(parametrization, values):
    for val, dict_val, hyperopt_val in values:
        parametrization.value = val
        assert flatten(_hp_parametrization_to_dict(parametrization)) == pytest.approx(flatten(dict_val))
        assert flatten(_hp_dict_to_parametrization(hyperopt_val)) == pytest.approx(
            flatten(parametrization.value)
        )
