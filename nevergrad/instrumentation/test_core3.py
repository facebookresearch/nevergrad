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
