# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import typing as t
import pytest
import numpy as np
from .core import Parameter
from . import parameter as par


def test_array_basics() -> None:
    var1 = par.Array(shape=(1,))
    var2 = par.Array(shape=(2, 2))
    d = par.Dict(var1=var1, var2=var2, var3=12)
    data = d.get_std_data()
    assert data.size == 5
    d.set_std_data(np.array([1, 2, 3, 4, 5]))
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


@pytest.mark.parametrize("param", [par.Dict(truc=12),  # type: ignore
                                   par.Tuple(), ])
def test_empty_parameters(param: par.Dict) -> None:
    assert not param.dimension
    assert not param.get_data_hash()
    if not param:  # Dict is not empty even though it is constant
        assert not param.get_value_hash()


def _true(*args: t.Any, **kwargs: t.Any) -> bool:  # pylint: disable=unused-argument
    return True


@pytest.mark.parametrize("param", [par.Array(shape=(2, 2)),  # type: ignore
                                   par.Array(init=np.ones(3)).set_mutation(sigma=3, exponent=5),
                                   par.Scalar(),
                                   par.Scalar(1.0).set_mutation(exponent=2.),
                                   par.Dict(blublu=par.Array(shape=(2, 3)), truc=12),
                                   par.Tuple(par.Array(shape=(2, 3)), 12),
                                   par.Instrumentation(par.Array(shape=(2,)), nonhash=[1, 2], truc=par.Array(shape=(1, 3))),
                                   par.Choice([par.Array(shape=(2,)), "blublu"]),
                                   par.TransitionChoice([par.Array(shape=(2,)), par.Scalar()]),
                                   ],
                         )
def test_parameters_basic_features(param: Parameter) -> None:
    seed = np.random.randint(2 ** 32, dtype=np.uint32)
    print(f"Seeding with {seed} from reproducibility.")
    np.random.seed(seed)
    assert isinstance(param.name, str)
    assert param._random_state is None
    assert param.generation == 0
    child = param.spawn_child()
    assert isinstance(child, type(param))
    assert child.generation == 1
    assert param._random_state is not None
    child.mutate()
    assert child.name == param.name
    assert child.random_state is param.random_state
    assert child.get_data_hash() != param.get_data_hash()
    assert child.uid != param.uid
    assert child.parents_uids == [param.uid]
    assert child.get_data_hash() != param.get_data_hash()  # Could be the same, for TransitionChoice with constants for instance
    child_hash = param.spawn_child()
    param.value = child.value
    assert param.get_value_hash() == child.get_value_hash()
    if isinstance(param, par.Array):
        assert param.get_value_hash() != child_hash.get_value_hash()
        child_hash.value = param.value
        assert param.get_data_hash() == child_hash.get_data_hash()
    param.recombine(child, child)
    # constraints
    param.register_cheap_constraint(_true)
    with pytest.warns(UserWarning):
        param.register_cheap_constraint(lambda *args, **kwargs: False)
    child2 = param.spawn_child()
    assert child.satisfies_constraint()
    assert not param.satisfies_constraint()
    assert not child2.satisfies_constraint()
    # array to and from with hash
    data_hash = param.get_data_hash()
    param.set_std_data(param.get_std_data())
    try:
        assert data_hash == param.get_data_hash()
    except AssertionError:
        # sometimes there can be some rounding errors...
        np.testing.assert_almost_equal(np.frombuffer(param.get_data_hash()), np.frombuffer(data_hash))
    # picklable
    string = pickle.dumps(child)
    pickle.loads(string)
    # array info transfer:
    if isinstance(param, par.Array):
        for name in ("integer", "exponent", "bounds", "bound_transform", "full_range_sampling"):
            assert getattr(param, name) == getattr(child, name)


@pytest.mark.parametrize(  # type: ignore
    "param,name",
    [(par.Array(shape=(2, 2)), "Array{(2,2)}[recombination=average,sigma=1.0]"),
     (par.Tuple(12), "Tuple(12)"),
     (par.Dict(constant=12), "Dict(constant=12)"),
     (par.Scalar(), "Scalar[recombination=average,sigma=Log{exp=1.2}[recombination=average,sigma=1.0]]"),
     (par.Scalar().set_integer_casting(), "Scalar{int}[recombination=average,sigma=Log{exp=1.2}[recombination=average,sigma=1.0]]"),
     (par.Instrumentation(par.Array(shape=(2,)), string="blublu", truc="plop"),
      "Instrumentation(Tuple(Array{(2,)}[recombination=average,sigma=1.0]),Dict(string=blublu,truc=plop))"),
     (par.Choice([1, 12]), "Choice(choices=Tuple(1,12),weights=Array{(2,)}[recombination=average,sigma=1.0])"),
     (par.Choice([1, 12], deterministic=True), "Choice{det}(choices=Tuple(1,12),weights=Array{(2,)}[recombination=average,sigma=1.0])"),
     (par.TransitionChoice([1, 12]), "TransitionChoice(choices=Tuple(1,12),position=Scalar[recombination=average,"
                                     "sigma=Log{exp=1.2}[recombination=average,sigma=1.0]],transitions=[1. 1.])")
     ]
)
def test_parameter_names(param: Parameter, name: str) -> None:
    assert param.name == name


@pytest.mark.parametrize(  # type: ignore
    "param,continuous,deterministic",
    [(par.Array(shape=(2, 2)), True, True),
     (par.Choice([True, False]), True, False),
     (par.Choice([True, False], deterministic=True), False, True),
     (par.Choice([True, par.Scalar().set_integer_casting()]), False, False),
     (par.Dict(constant=12, data=par.Scalar().set_integer_casting()), False, True),
     ]
)
def test_parameter_descriptors(param: Parameter, continuous: bool, deterministic: bool) -> None:
    assert param.descriptors.continuous == continuous
    assert param.descriptors.deterministic == deterministic


def test_instrumentation() -> None:
    inst = par.Instrumentation(par.Array(shape=(2,)), string="blublu", truc=par.Array(shape=(1, 3)))
    inst.mutate()
    assert len(inst.args) == 1
    assert len(inst.kwargs) == 2


def test_scalar_and_mutable_sigma() -> None:
    param = par.Scalar(init=1.0, mutable_sigma=True).set_mutation(exponent=2.0, sigma=5)
    assert param.value == 1
    data = param.get_std_data()
    assert data[0] == 0.0
    param.set_std_data(np.array([-0.2]))
    assert param.value == 0.5
    assert param.sigma.value == 5
    param.mutate()
    assert param.sigma.value != 5
    param.set_integer_casting()
    assert isinstance(param.value, int)


def test_array_recombination() -> None:
    param = par.Tuple(par.Scalar(1.0, mutable_sigma=True).set_mutation(sigma=5))
    param2 = par.Tuple(par.Scalar(1.0, mutable_sigma=True).set_mutation(sigma=1))
    param.value = (1,)
    param2.value = (3,)
    param.recombine(param2)
    assert param.value[0] == 2.0
    param2.set_std_data((param.get_std_data() + param2.get_std_data()) / 2)
    assert param2.value[0] == 1.7  # because of different sigma, this is not the "expected" value


@pytest.mark.parametrize(  # type: ignore
    "name", ["clipping", "arctan", "tanh", "constraint"]
)
def test_constraints(name: str) -> None:
    param = par.Scalar(12.0).set_mutation(sigma=2).set_bounds(method=name, a_min=-100, a_max=100)
    param.set_std_data(param.get_std_data())
    np.testing.assert_approx_equal(param.value, 12, err_msg="Back and forth did not work")
    param.set_std_data(np.array([100000.0]))
    if param.satisfies_constraint():
        np.testing.assert_approx_equal(param.value, 100, significant=3, err_msg="Constraining did not work")


@pytest.mark.parametrize(  # type: ignore
    "param,expected", [(par.Scalar(), False), (par.Scalar().set_bounds(-1000, 1000, full_range_sampling=True), True)]
)
def test_scalar_sampling(param: par.Scalar, expected: bool) -> None:
    assert not any(np.abs(param.spawn_child().value) > 100 for _ in range(10))
    assert any(np.abs(param.sample().value) > 100 for _ in range(10)) == expected


def test_log() -> None:
    with pytest.warns(UserWarning) as record:
        par.Log(a_min=0.001, a_max=0.1, init=0.01, exponent=2.0)
        assert not record
        par.Log(a_min=0.001, a_max=0.1, init=0.01, exponent=10.0)
        assert len(record) == 1


def test_ordered_choice() -> None:
    choice = par.TransitionChoice([0, 1, 2, 3], transitions=[-1000000, 10])
    assert choice.value == 2
    choice.value = 1
    assert choice.value == 1
    choice.mutate()
    assert choice.value in [0, 2]
    assert choice.get_std_data().size
    choice.set_std_data(np.array([12.0]))
    assert choice.value == 3
