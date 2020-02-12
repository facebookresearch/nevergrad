import nevergrad as ng


def test_parametrization() -> None:
    # DOC_PARAM_0
    arg1 = ng.p.TransitionChoice(["a", "b"])
    arg2 = ng.p.Choice(["a", "c", "e"])
    value = ng.p.Scalar()

    instru = ng.p.Instrumentation(arg1, arg2, "blublu", value=value)
    print(instru.dimension)
    # >>> 5
    # DOC_PARAM_1
    def myfunction(arg1, arg2, arg3, value=3):
        print(arg1, arg2, arg3)
        return value**2

    optimizer = ng.optimizers.OnePlusOne(parametrization=instru, budget=100)
    recommendation = optimizer.minimize(myfunction)
    print(recommendation.value)
    # >>> (('b', 'e', 'blublu'), {'value': -0.00014738768964717153})
    # DOC_PARAM_2
    instru2 = instru.spawn_child().set_standardized_data([1, -80, -80, 80, 3])
    assert instru2.args == ('b', 'e', 'blublu')
    assert instru2.kwargs == {'value': 3}
    # DOC_PARAM_3
