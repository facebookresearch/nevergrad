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
                                   par.NgList(), ])
def test_empty_parameters(param: Parameter) -> None:
    assert not param.dimension
    assert not param.get_data_hash()
    assert not param.get_value_hash()


@pytest.mark.parametrize("param", [par.Array((2, 2), sigma=2),  # type: ignore
                                   par.NgDict(blublu=par.Array((2, 3)), truc=12),
                                   par.NgList(par.Array((2, 3)), 12), ])
def test_parameters_basic_features(param: Parameter) -> None:
    assert isinstance(param.name, str)
    assert param._random_state is None
    assert param.generation == 0
    child = param.spawn_child()
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


def test_choices() -> None:
    param1 = par.Array((2, 2), sigma=2)
    param2 = par.Array((2,), sigma=1)
    choice = par.Choice([param1, param2, "blublu"])
    choice.value = "blublu"
    np.testing.assert_array_almost_equal(choice.probabilities.value, [0, 0, 0.69314718])
