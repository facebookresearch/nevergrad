# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
import numpy as np
import nevergrad as ng

# pylint: disable=reimported,redefined-outer-name,unused-variable,unsubscriptable-object, unused-argument
# pylint: disable=import-outside-toplevel
# black tends to make too long lines for the doc
# fmt: off


def test_simplest_example() -> None:
    # DOC_SIMPLEST_0
    import nevergrad as ng

    def square(x):
        return sum((x - 0.5) ** 2)

    # optimization on x as an array of shape (2,)
    optimizer = ng.optimizers.NGOpt(parametrization=2, budget=100)
    recommendation = optimizer.minimize(square)  # best value
    print(recommendation.value)
    # >>> [0.49999998 0.50000004]
    # DOC_SIMPLEST_1
    np.testing.assert_array_almost_equal(recommendation.value, [0.5, 0.5], decimal=1)


# pylint: disable=function-redefined
def test_base_example() -> None:
    # DOC_BASE_0
    import nevergrad as ng

    def square(x, y=12):
        return sum((x - 0.5) ** 2) + abs(y)

    # optimization on x as an array of shape (2,)
    optimizer = ng.optimizers.NGOpt(parametrization=2, budget=100)
    recommendation = optimizer.minimize(square)  # best value
    print(recommendation.value)
    # >>> [0.49971112 0.5002944 ]
    # DOC_BASE_1
    instrum = ng.p.Instrumentation(
        ng.p.Array(shape=(2,)).set_bounds(lower=-12, upper=12),
        y=ng.p.Scalar()
    )
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100)
    recommendation = optimizer.minimize(square)
    print(recommendation.value)
    # >>> ((array([0.52213095, 0.45030925]),), {'y': -0.0003603100877068604})
    # DOC_BASE_2
    from concurrent import futures

    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=10, num_workers=2)

    with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
        recommendation = optimizer.minimize(square, executor=executor, batch_mode=False)
    # DOC_BASE_3
    import nevergrad as ng

    def square(x, y=12):
        return sum((x - 0.5) ** 2) + abs(y)

    instrum = ng.p.Instrumentation(ng.p.Array(shape=(2,)), y=ng.p.Scalar())  # We are working on R^2 x R.
    optimizer = ng.optimizers.NGOpt(parametrization=instrum, budget=100, num_workers=1)

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        loss = square(*x.args, **x.kwargs)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    print(recommendation.value)
    # DOC_BASE_4
    import nevergrad as ng

    def onemax(x):
        return len(x) - x.count(1)

    # Discrete, ordered
    param = ng.p.TransitionChoice(range(7), repetitions=10)
    optimizer = ng.optimizers.DiscreteOnePlusOne(parametrization=param, budget=100, num_workers=1)

    recommendation = optimizer.provide_recommendation()
    for _ in range(optimizer.budget):
        x = optimizer.ask()
        # loss = onemax(*x.args, **x.kwargs)  # equivalent to x.value if not using Instrumentation
        loss = onemax(x.value)
        optimizer.tell(x, loss)

    recommendation = optimizer.provide_recommendation()
    print(recommendation.value)
    # >>> (1, 1, 0, 1, 1, 4, 1, 1, 1, 1)
    # DOC_BASE_5


def test_print_all_optimizers() -> None:
    # DOC_OPT_REGISTRY_0
    import nevergrad as ng

    print(sorted(ng.optimizers.registry.keys()))
    # DOC_OPT_REGISTRY_1


def test_parametrization() -> None:
    # DOC_PARAM_0
    arg1 = ng.p.Choice(["Helium", "Nitrogen", "Oxygen"])
    arg2 = ng.p.TransitionChoice(["Solid", "Liquid", "Gas"])
    values = ng.p.Tuple(ng.p.Scalar().set_integer_casting(), ng.p.Scalar())

    instru = ng.p.Instrumentation(arg1, arg2, "blublu", amount=values)
    print(instru.dimension)
    # >>> 6
    # DOC_PARAM_1

    def myfunction(arg1, arg2, arg3, amount=(2, 2)):
        print(arg1, arg2, arg3)
        return amount[0] ** 2 + amount[1] ** 2

    optimizer = ng.optimizers.NGOpt(parametrization=instru, budget=100)
    recommendation = optimizer.minimize(myfunction)
    print(recommendation.value)
    # >>> (('Helium', 'Gas', 'blublu'), {'value': (0, 0.0006602471804655007)})
    # DOC_PARAM_2
    instru2 = instru.spawn_child().set_standardized_data([-80, 80, -80, 0, 3, 5.0])
    assert instru2.args == ("Nitrogen", "Liquid", "blublu")
    assert instru2.kwargs == {"amount": (3, 5.0)}
    # DOC_PARAM_3


def test_doc_constrained_optimization() -> None:
    np.random.seed(12)  # avoid flakiness
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        # DOC_CONSTRAINED_0
        import nevergrad as ng

        def square(x):
            return sum((x - 0.5) ** 2)

        optimizer = ng.optimizers.NGOpt(parametrization=2, budget=100)
        # define a constraint on first variable of x:
        optimizer.parametrization.register_cheap_constraint(lambda x: x[0] >= 1)

        recommendation = optimizer.minimize(square, verbosity=2)
        print(recommendation.value)
        # >>> [1.00037625, 0.50683314]
        # DOC_CONSTRAINED_1
    np.testing.assert_array_almost_equal(recommendation.value, [1, 0.5], decimal=1)


def test_callback_doc() -> None:
    # DOC_CALLBACK_0
    import nevergrad as ng

    def my_function(x):
        return abs(sum(x - 1))

    def print_candidate_and_value(optimizer, candidate, value):
        print(candidate, value)

    optimizer = ng.optimizers.NGOpt(parametrization=2, budget=4)
    optimizer.register_callback("tell", print_candidate_and_value)
    optimizer.minimize(my_function)  # triggers a print at each tell within minimize
    # DOC_CALLBACK_1


def test_inoculation() -> None:
    # DOC_INOCULATION_0
    optim = ng.optimizers.NGOpt(parametrization=2, budget=100)
    optim.suggest([12, 12])
    candidate = optim.ask()
    # equivalent to:
    candidate = optim.parametrization.spawn_child(new_value=[12, 12])
    # you can then use to tell the loss
    optim.tell(candidate, 2.0)
    # DOC_INOCULATION_1
    param = ng.p.Instrumentation(ng.p.Choice(["a", "b", "c"]), lr=ng.p.Log(lower=0.001, upper=1.0))
    optim = ng.optimizers.NGOpt(parametrization=param, budget=100)
    optim.suggest("c", lr=0.02)
    candidate = optim.ask()
    # equivalent to:
    candidate = optim.parametrization.spawn_child(new_value=(("c",), {"lr": 0.02}))
    # you can then use to tell the loss
    optim.tell(candidate, 2.0)
    # DOC_INOCULATION_2
