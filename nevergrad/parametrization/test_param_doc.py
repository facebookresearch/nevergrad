# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
# pylint: disable=reimported,redefined-outer-name,unused-variable,unsubscriptable-object, unused-argument
# pylint: disable=import-outside-toplevel


def test_readme_parametrization() -> None:
    # @@@ IF YOU CHANGE THIS EXAMPLE @@@
    # please change it too in README.md

    # DOC_README_0
    import nevergrad as ng

    def fake_training(learning_rate: float, batch_size: int, architecture: str) -> float:
        # optimal for learning_rate=0.2, batch_size=4, architecture="conv"
        return (learning_rate - 0.2)**2 + (batch_size - 4)**2 + (0 if architecture == "conv" else 10)

    # Instrumentation class is used for functions with multiple inputs
    # (positional and/or keywords)
    parametrization = ng.p.Instrumentation(
        # a log-distributed scalar between 0.001 and 1.0
        learning_rate=ng.p.Log(lower=0.001, upper=1.0),
        # an integer from 1 to 12
        batch_size=ng.p.Scalar(lower=1, upper=12).set_integer_casting(),
        # either "conv" or "fc"
        architecture=ng.p.Choice(["conv", "fc"])
    )

    optimizer = ng.optimizers.OnePlusOne(parametrization=parametrization, budget=100)
    recommendation = optimizer.minimize(fake_training)

    print(recommendation.kwargs)  # shows the recommended keyword arguments of the function
    # >>> {'learning_rate': 0.1998, 'batch_size': 4, 'architecture': 'conv'}
    # DOC_README_1


def test_param_example() -> None:
    # DOC_PARAM_0
    import nevergrad as ng

    # build a parameter providing a dict value:
    param = ng.p.Dict(
        # logarithmically distributed float
        log=ng.p.Log(lower=0.01, upper=1.0),
        # one-dimensional array of length 2
        array=ng.p.Array(shape=(2,)),
        # character, either "a" or "b or "c".
        char=ng.p.Choice(["a", "b", "c"])
    )

    print(param.value)
    # {'log': 0.01,
    #  'array': array([0., 0.]),
    #  'char': 'a'}
    # DOC_PARAM_1

    # DOC_PARAM_10
    # create a new instance
    child = param.spawn_child()
    # update its value
    child.value = {'log': 0.2,
                   'array': np.array([12., 13.]),
                   'char': 'c'}

    # export to standardized space
    data = child.get_standardized_data(reference=param)
    print(data)
    # np.array([12., 13.,  0.,  0., 0.69, 0.90])
    # DOC_PARAM_11

    # DOC_PARAM_100
    param.mutate()
    print(param.value)
    # {'log': 0.155,
    #  'array': np.array([-0.966, 0.045]),
    #  'char': 'a'}

    # increase the step/sigma for array
    # (note that it's advised to to this during the creation
    #  of the variable:
    #  array=ng.p.Array(shape=(2,)).set_mutation(sigma=10))
    param["array"].set_mutation(sigma=10)  # type: ignore
    param.mutate()
    print(param.value)
    # {'log': 0.155,
    #  'array': np.array([-9.47, 8.38]),  # larger mutation
    #  'char': 'a'}

    # DOC_PARAM_101

    # DOC_PARAM_1000
    param.random_state.seed(12)
    # DOC_PARAM_1001
