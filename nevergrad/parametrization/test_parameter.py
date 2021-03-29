# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as tp
import pytest
import numpy as np

from nevergrad.common import errors
from . import parameter as par


def test_array_basics() -> None:
    var1 = par.Array(shape=(1,))
    var2 = par.Array(shape=(2, 2))
    d = par.Dict(var1=var1, var2=var2, var3=12)
    data = d.get_standardized_data(reference=d)
    assert data.size == 5
    d.set_standardized_data(np.array([1, 2, 3, 4, 5]))
    assert var1.value[0] == 1
    np.testing.assert_array_equal(d.value["var2"], np.array([[2, 3], [4, 5]]))
    # setting value on arrays
    with pytest.raises(ValueError):
        var1.value = np.array([1, 2])
    with pytest.raises(TypeError):
        var1.value = 4  # type: ignore
    var1.value = np.array([2])
    representation = repr(d)
    assert "Dict(var1" in representation
    d.set_name("blublu")
    representation = repr(d)
    assert "blublu:{'var1" in representation


@pytest.mark.parametrize(  # type: ignore
    "param",
    [
        par.Dict(truc=12),
        par.Tuple(),
        par.Instrumentation(12),
    ],
)
def test_empty_parameters(param: par.Dict) -> None:
    assert not param.dimension
    analysis = par.helpers.analyze(param)
    assert analysis.continuous
    assert analysis.deterministic
    assert param.descriptors.continuous
    assert param.descriptors.deterministic


def _true(*args: tp.Any, **kwargs: tp.Any) -> bool:  # pylint: disable=unused-argument
    return True


@pytest.mark.parametrize(  # type: ignore
    "param",
    [
        par.Array(shape=(2, 2)),
        par.Array(init=np.ones(3)).set_mutation(sigma=3, exponent=5),
        par.Scalar(),
        par.Scalar(1.0).set_mutation(exponent=2.0),
        par.Dict(blublu=par.Array(shape=(2, 3)), truc=12),
        par.Dict(scalar=par.Scalar(), const_array=np.array([12.0, 12.0]), const_list=[3, 3]),
        par.Tuple(par.Array(shape=(2, 3)), 12),
        par.Instrumentation(par.Array(shape=(2,)), nonhash=[1, 2], truc=par.Array(shape=(1, 3))),
        par.Choice([par.Array(shape=(2,)), "blublu"]),
        par.Choice([1, 2], repetitions=2),
        par.TransitionChoice([par.Array(shape=(2,)), par.Scalar()]),
        par.TransitionChoice(["a", "b", "c"], transitions=(0, 2, 1), repetitions=4),
    ],
)
def test_parameters_basic_features(param: par.Parameter) -> None:
    check_parameter_features(param)
    check_parameter_freezable(param)


# pylint: disable=too-many-statements
def check_parameter_features(param: par.Parameter) -> None:
    seed = np.random.randint(2 ** 32, dtype=np.uint32)
    print(f"Seeding with {seed} from reproducibility.")
    np.random.seed(seed)
    assert isinstance(param.name, str)
    assert param._random_state is None
    assert param.generation == 0
    child = param.spawn_child()
    assert isinstance(child, type(param))
    assert child.heritage["lineage"] == param.uid
    assert child.generation == 1
    assert not np.any(param.get_standardized_data(reference=param))
    assert not np.any(child.get_standardized_data(reference=child))
    assert not np.any(child.get_standardized_data(reference=param))
    assert child.name == param.name
    assert param._random_state is not None
    assert child.random_state is param.random_state
    assert child.uid != param.uid
    assert child.parents_uids == [param.uid]
    mutable = True
    try:
        child.mutate()
    except par.NotSupportedError:
        mutable = False
    else:
        assert np.any(child.get_standardized_data(reference=param))
    param.set_name("blublu")
    child_hash = param.spawn_child()
    assert child_hash.name == "blublu"
    param.value = child.value
    try:
        assert param.get_value_hash() == child.get_value_hash()
    except AssertionError:
        # with approximations, value hash may return something slightly different
        # so let's try a relaxed version
        param2 = param.spawn_child()
        param2.value = child.value
        assert param2.get_value_hash() == param.get_value_hash()
    if isinstance(param, par.Data):
        assert param.get_value_hash() != child_hash.get_value_hash()
        child_hash.value = param.value
        np.testing.assert_almost_equal(param.get_standardized_data(reference=child), [0] * param.dimension)
    if mutable:
        param.recombine(child, child)
        param.recombine()  # empty should work, for simplicity's sake
    # constraints
    param.register_cheap_constraint(_true)
    with pytest.warns(UserWarning):
        param.register_cheap_constraint(lambda *args, **kwargs: False)
    child2 = param.spawn_child(param.value)  # just checking new_value
    assert child.satisfies_constraints()
    assert not param.satisfies_constraints()
    assert not child2.satisfies_constraints()
    # array to and from with hash
    data = param.get_standardized_data(reference=child2)
    param.set_standardized_data(data, reference=child2)
    np.testing.assert_array_almost_equal(param.get_standardized_data(reference=child2), data)
    # picklable
    string = pickle.dumps(child)
    pickle.loads(string)
    # sampling
    samp_param = param.sample()
    assert samp_param.uid == samp_param.heritage["lineage"]
    # set descriptor
    assert param.descriptors.deterministic_function
    assert param.function.deterministic
    param.descriptors.deterministic_function = False
    assert not param.descriptors.deterministic_function
    assert not param.function.deterministic
    #
    assert param.descriptors.non_proxy_function
    assert not param.function.proxy
    param.descriptors.non_proxy_function = False
    assert not param.descriptors.non_proxy_function
    assert param.function.proxy
    #
    descr_child = param.spawn_child()
    assert not descr_child.descriptors.deterministic_function
    assert not descr_child.descriptors.non_proxy_function


def check_parameter_freezable(param: par.Parameter) -> None:
    param.freeze()
    value = param.value
    data = param.get_standardized_data(reference=param)
    child = param.spawn_child()
    child.mutate()
    child.recombine(param)
    with pytest.raises(RuntimeError):
        param.value = value
    with pytest.raises(RuntimeError):
        param.set_standardized_data(data)
    child.set_standardized_data(data, reference=param)
    with pytest.raises(RuntimeError):
        param.recombine(child)


@pytest.mark.parametrize(  # type: ignore
    "param,name",
    [
        (par.Array(shape=(2, 2)), "Array{(2,2)}"),
        (par.Tuple(12), "Tuple(12)"),
        (par.Dict(constant=12), "Dict(constant=12)"),
        (par.Scalar(), "Scalar[sigma=Scalar{exp=2.03}]"),
        (
            par.Log(lower=3.2, upper=12.0, exponent=1.5),
            "Log{Cl(2.868682869489701,6.128533874054364,b),exp=1.50}",
        ),
        (par.Scalar().set_integer_casting(), "Scalar{Int}[sigma=Scalar{exp=2.03}]"),
        (
            par.Instrumentation(par.Array(shape=(2,)), string="blublu", truc="plop"),
            "Instrumentation(Tuple(Array{(2,)}),Dict(string=blublu,truc=plop))",
        ),
        (par.Choice([1, 12]), "Choice(choices=Tuple(1,12),indices=Array{(1,2),SoftmaxSampling})"),
        (
            par.Choice([1, 12], deterministic=True),
            "Choice(choices=Tuple(1,12),indices=Array{(1,2),SoftmaxSampling{det}})",
        ),
        (
            par.TransitionChoice([1, 12]),
            "TransitionChoice(choices=Tuple(1,12),indices=Array{Cd(0,2),Add,Int},transitions=[1. 1.])",
        ),
    ],
)
def test_parameter_names(param: par.Parameter, name: str) -> None:
    assert param.name == name


@pytest.mark.parametrize(  # type: ignore
    "param,continuous,deterministic,ordered",
    [
        (par.Array(shape=(2, 2)), True, True, True),
        (par.Choice([True, False]), True, False, False),
        (par.Choice([True, False], deterministic=True), False, True, False),
        (par.Choice([True, par.Scalar().set_integer_casting()]), False, False, False),
        (
            par.Dict(constant=12, data=par.Scalar().set_integer_casting()),
            False,
            True,
            True,
        ),
    ],
)
def test_parameter_analysis(
    param: par.Parameter, continuous: bool, deterministic: bool, ordered: bool
) -> None:
    analysis = par.helpers.analyze(param)
    assert analysis.continuous == continuous
    assert analysis.deterministic == deterministic
    assert analysis.ordered == ordered
    assert param.descriptors.continuous == continuous
    assert param.descriptors.deterministic == deterministic
    assert param.descriptors.ordered == ordered


def test_instrumentation() -> None:
    inst = par.Instrumentation(par.Array(shape=(2,)), string="blublu", truc=par.Array(shape=(1, 3)))
    inst.mutate()
    assert len(inst.args) == 1
    assert len(inst.kwargs) == 2
    scal = par.Scalar()
    with pytest.raises(ValueError):
        inst = par.Instrumentation(scal, blublu=scal)
    assert set(inst[1]) == {"string", "truc"}  # type: ignore


def test_scalar_and_mutable_sigma() -> None:
    param = par.Scalar(init=1.0, mutable_sigma=True).set_mutation(exponent=2.0, sigma=5)
    assert param.value == 1
    data = param.get_standardized_data(reference=param)
    assert data[0] == 0.0
    param.set_standardized_data(np.array([-0.2]))
    assert param.value == 0.5
    assert param.sigma.value == pytest.approx(5)
    param.mutate()
    assert param.sigma.value != pytest.approx(5)
    param.set_integer_casting()
    assert isinstance(param.value, int)


def test_array_recombination() -> None:
    param = par.Tuple(par.Scalar(1.0, mutable_sigma=True).set_mutation(sigma=5))
    param2 = par.Tuple(par.Scalar(1.0, mutable_sigma=True).set_mutation(sigma=1))
    param.value = (1,)
    param2.value = (3,)
    param.recombine(param2)
    assert param.value[0] == 2.0
    param2.set_standardized_data(
        (param.get_standardized_data(reference=param2) + param2.get_standardized_data(reference=param2)) / 2
    )
    assert param2.value[0] == 2.5


def _false(value: tp.Any) -> bool:  # pylint: disable=unused-argument
    return False


def test_endogeneous_constraint() -> None:
    param = par.Scalar(1.0, mutable_sigma=True)
    param.sigma.register_cheap_constraint(_false)
    assert not param.satisfies_constraints()


def _return_val(val: float) -> float:  # pylint: disable=unused-argument
    return val


@pytest.mark.parametrize("val,expected", [(1.0, True), (0.0, True), (-1.0, False)])  # type: ignore
def test_float_constraint(val: float, expected: bool) -> None:
    param = par.Scalar(val, mutable_sigma=True)
    param.register_cheap_constraint(_return_val)
    assert param.satisfies_constraints() is expected


@pytest.mark.parametrize("name", ["clipping", "arctan", "tanh", "constraint", "bouncing"])  # type: ignore
def test_constraints(name: str) -> None:
    param = par.Scalar(12.0).set_mutation(sigma=2).set_bounds(method=name, lower=-100, upper=100)
    param.set_standardized_data(param.get_standardized_data(reference=param))
    np.testing.assert_approx_equal(param.value, 12, err_msg="Back and forth did not work")
    param.set_standardized_data(np.array([100000.0]))
    if param.satisfies_constraints():
        # bouncing works differently from others
        np.testing.assert_approx_equal(
            param.value,
            100 if name != "bouncing" else -100,
            significant=3,
            err_msg="Constraining did not work",
        )


@pytest.mark.parametrize(  # type: ignore
    "param,expected",
    [
        (par.Scalar(), False),
        (par.Scalar(lower=-1000, upper=1000).set_mutation(sigma=1), True),
        (par.Scalar(lower=-1000, upper=1000, init=0).set_mutation(sigma=1), False),
        (par.Scalar().set_bounds(-1000, 1000, full_range_sampling=True), True),
    ],
)
def test_scalar_sampling(param: par.Scalar, expected: bool) -> None:
    spawned_vals = [np.abs(param.spawn_child().value) for _ in range(10)]
    sampled_vals = [np.abs(param.sample().value) for _ in range(10)]
    assert not np.any(np.array(spawned_vals) > 100)
    assert np.any(np.array(sampled_vals) > 100) == expected


def test_log() -> None:
    with pytest.warns(errors.NevergradRuntimeWarning) as record:
        log = par.Log(lower=0.001, upper=0.1, init=0.02, exponent=2.0)
        assert log.value == pytest.approx(0.02)
        assert not record, [x.message for x in record]  # TODO readd
        par.Log(lower=0.001, upper=0.1, init=0.01, exponent=10.0)
        assert len(record) == 1, [x.message for x in record]
    # automatic
    log = par.Log(lower=0.001, upper=0.1)
    assert log.value == pytest.approx(0.01)
    log.set_standardized_data([2.999])
    np.testing.assert_almost_equal(log.value, 0.09992, decimal=5)


def test_bounded_scalar() -> None:
    scalar = par.Scalar(lower=0.0, upper=0.6)
    np.testing.assert_almost_equal(scalar.sigma.value, 0.1)
    np.testing.assert_almost_equal(scalar.value, 0.3)
    # partial
    with pytest.raises(ValueError):
        scalar = par.Scalar(lower=1.0)


def test_ordered_choice() -> None:
    choice = par.TransitionChoice([0, 1, 2, 3], transitions=[-1000000, 10])
    assert len(choice) == 4
    assert choice.value == 2
    choice.value = 1
    assert choice.value == 1
    choice.mutate()
    assert choice.value in [0, 2]
    assert choice.get_standardized_data(reference=choice).size
    choice.set_standardized_data(np.array([12.0]))
    assert choice.value == 3


def test_ordered_choice_weird_values() -> None:
    choice = par.TransitionChoice([0, np.nan, np.inf])
    choice.value = np.nan
    assert choice.value is np.nan
    choice.value = np.inf
    assert choice.value == np.inf


def test_choice_repetitions() -> None:
    choice = par.Choice([0, 1, 2, 3], repetitions=2)
    choice.random_state.seed(12)
    assert len(choice) == 4
    assert choice.value == (0, 2)
    choice.value = (3, 1)
    assert choice.indices.value.tolist() == [3, 1]
    choice.mutate()


def test_transition_choice_repetitions() -> None:
    choice = par.TransitionChoice([0, 1, 2, 3], repetitions=2)
    choice.random_state.seed(12)
    assert len(choice) == 4
    assert choice.value == (2, 2)
    choice.value = (3, 1)
    np.testing.assert_almost_equal(choice.indices.value, [3, 1], decimal=3)
    choice.mutate()
    assert choice.value == (3, 0)


def test_array_bounded_initialization() -> None:
    array = par.Array(shape=(1,), lower=-1)
    assert array.value[0] == 0
    assert array.bounds == (-1, None)  # type: ignore
    assert array.sigma.value == 1.0
    array = par.Array(shape=(1,), lower=-0.5, upper=2.5)
    assert array.value[0] == 1
    assert array.bounds == (-0.5, 2.5)  # type: ignore
    assert array.sigma.value == 0.5


@pytest.mark.parametrize("method", ["clipping", "arctan", "tanh", "constraint", "bouncing"])  # type: ignore
@pytest.mark.parametrize("exponent", [2.0, None])  # type: ignore
@pytest.mark.parametrize("sigma", [1.0, 1000, 0.001])  # type: ignore
def test_array_sampling(method: str, exponent: tp.Optional[float], sigma: float) -> None:
    mbound = 10000
    param = par.Array(init=2 * np.ones((2, 3))).set_bounds(
        [1, 1, 1], [mbound] * 3, method=method, full_range_sampling=True
    )
    param.set_mutation(exponent=exponent, sigma=sigma)
    new_param = param.sample()
    val = new_param.value
    assert np.any(np.abs(val) > 10)
    assert np.all(val <= mbound)
    assert np.all(val >= 1)


def test_parenthood() -> None:
    param = par.Instrumentation(par.Scalar(init=1.0, mutable_sigma=True).set_mutation(exponent=2.0, sigma=5))
    sigma_uid = param[0][0].sigma.uid  # type: ignore
    param_samp = param.sample()
    param_spawn = param.spawn_child()
    assert param_samp[0][0].sigma.parents_uids == []  # type: ignore
    assert param_spawn[0][0].sigma.parents_uids == [sigma_uid]  # type: ignore
