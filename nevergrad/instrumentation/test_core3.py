import pytest
import numpy as np
from . import core3


def test_array_basics() -> None:
    var1 = core3.Array(1)
    var2 = core3.Array(2, 2)
    d = core3.ParametersDict(var1=var1, var2=var2)
    data = d.to_std_data()
    assert data.size == 5
    d.with_std_data(np.array([1, 2, 3, 4, 5]))
    assert var1.value[0] == 1
    np.testing.assert_array_equal(d.value["var2"], np.array([[2, 3], [4, 5]]))
    # setting value on arrays
    with pytest.raises(ValueError):
        var1.value = np.array([1, 2])
    with pytest.raises(TypeError):
        var1.value = 4  # type: ignore
    var1.value = np.array([2])
    representation = repr(d)
    assert "ParametersDict{var1" in representation
    d.with_name("blublu")
    representation = repr(d)
    assert "blublu:{'var1" in representation
