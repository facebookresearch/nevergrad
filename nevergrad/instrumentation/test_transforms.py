from typing import List
import numpy as np
from ..common import testing
from . import transforms


@testing.parametrized(
    affine=(transforms.Affine(3, 4), "Af(3,4)"),
    reshape=(transforms.Reshape(4, 3), "Rs(4,3)"),
    exponentiate=(transforms.Exponentiate(3, 4), "Ex(3,4)"),
    tanh=(transforms.TanhBound(3, 4), "Th(3,4)"),
    arctan=(transforms.ArctanBound(3, 4), "At(3,4)"),
)
def test_back_and_forth(transform: transforms.Transform, string: str) -> None:
    x = np.random.normal(0, 1, size=12)
    y = transform.forward(x)
    x2 = transform.backward(y)
    np.testing.assert_array_almost_equal(x2, x)
    np.testing.assert_equal(f"{transform:short}", string)
    print(f"{transform}")


@testing.parametrized(
    affine=(transforms.Affine(3, 4), [0, 1, 2], [4, 7, 10]),
    reshape=(transforms.Reshape(2, 3), [0, 1, 2, 3, 4, 5], [[0, 1, 2], [3, 4, 5]]),
    exponentiate=(transforms.Exponentiate(10, -1.), [0, 1, 2], [1, .1, .01]),
    tanh=(transforms.TanhBound(3, 5), [-100000, 100000, 0], [3, 5, 4]),
    arctan=(transforms.ArctanBound(3, 5), [-100000, 100000, 0], [3, 5, 4]),
)
def test_vals(transform: transforms.Transform, x: List[float], expected: List[float]) -> None:
    y = transform.forward(np.array(x))
    np.testing.assert_almost_equal(y, expected, decimal=5)
