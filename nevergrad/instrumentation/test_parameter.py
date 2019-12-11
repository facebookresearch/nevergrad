# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import pytest
import numpy as np
from .core3 import Parameter
from . import parameter as par


def test_array_basics() -> None:
    var1 = par.Array((1,))
    var2 = par.Array((2, 2))
    d = par.NgDict(var1=var1, var2=var2, var3=12)
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
    assert "NgDict{var1" in representation
    d.set_name("blublu")
    representation = repr(d)
    assert "blublu:{'var1" in representation


@pytest.mark.parametrize("param", [par.NgDict(truc=12),  # type: ignore
                                   par.NgTuple(), ])
def test_empty_parameters(param: Parameter) -> None:
    assert not param.dimension
    assert not param.get_data_hash()
    assert not param.get_value_hash()


@pytest.mark.parametrize("param", [par.Array((2, 2)),  # type: ignore
                                   par.Array((3,)).set_mutation(sigma=3, exponent=5),
                                   par.Scalar(),
                                   par.Scalar().set_mutation(exponent=2.),  # should bug so far (exponent not propagated)
                                   par.NgDict(blublu=par.Array((2, 3)), truc=12),
                                   par.NgTuple(par.Array((2, 3)), 12),
                                   par.Instrumentation(par.Array((2,)), string="blublu", truc=par.Array((1, 3))),
                                   par.Choice([par.Array((2,)), "blublu"])])
def test_parameters_basic_features(param: Parameter) -> None:
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
    assert child.get_data_hash() != param.get_data_hash()
    param.value = child.value
    assert param.get_value_hash() == child.get_value_hash()
    if isinstance(param, par.Array):
        assert param.get_data_hash() == child.get_data_hash()
    param.recombine(child, child)
    # constraints
    param.register_cheap_constraint(lambda x: False)
    child2 = param.spawn_child()
    assert child.complies_with_constraint()
    assert not param.complies_with_constraint()
    assert not child2.complies_with_constraint()
    # array to and from with hash
    data_hash = param.get_data_hash()
    param.set_std_data(param.get_std_data())
    assert data_hash == param.get_data_hash()
    # picklable
    string = pickle.dumps(child)
    pickle.loads(string)
    # array info transfer:
    if isinstance(param, par.Array):
        for name in ("exponent", "bounds", "bounding_method", "full_range_sampling"):
            assert getattr(param, name) == getattr(child, name)


def test_choices() -> None:
    param1 = par.Array((2, 2)).set_mutation(sigma=2.0)
    param2 = par.Array((2,))
    choice = par.Choice([param1, param2, "blublu"])
    choice.value = "blublu"
    np.testing.assert_array_almost_equal(choice.probabilities.value, [0, 0, 0.69314718])
    choice.probabilities.value = np.array([1000.0, 0, 0])
    choice.mutate()
    assert np.abs(choice.choices[0].value).ravel().sum()
    assert not np.abs(choice.choices[1].value).ravel().sum(), "Only selection should mutate"
    with pytest.raises(ValueError):
        choice.value = "hop"
    choice.value = np.array([1, 1])
    np.testing.assert_array_almost_equal(choice.probabilities.value, [0, 0.69314718, 0])


def test_instrumentation() -> None:
    inst = par.Instrumentation(par.Array((2,)), string="blublu", truc=par.Array((1, 3)))
    inst.mutate()
    assert len(inst.args) == 1
    assert len(inst.kwargs) == 2


def test_scalar() -> None:
    param = par.Scalar().set_mutation(exponent=2., sigma=5)
    assert param.value == 1
    data = param.get_std_data()
    assert data[0] == 0.0
    param.set_std_data(np.array([-0.2]))
    assert param.value == 0.5


def test_log() -> None:
    with pytest.warns(UserWarning) as record:
        par.Log(0.001, 0.1, init=0.01, exponent=2)
        assert not record
        par.Log(0.001, 0.1, init=0.01, exponent=10)
        assert len(record) == 1
