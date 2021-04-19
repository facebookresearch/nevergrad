# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Experimental mutation patterns for structured data
"""

import typing as tp
import numpy as np
from . import core
from . import transforms
from .data import Data

# from .choice import Choice
from . import _layering


D = tp.TypeVar("D", bound=Data)


class Mutation(_layering.Layered):
    """Custom mutation or recombination
    This is an experimental API

    Either implement:
    - `_apply_array`  which provides a new np.ndarray from a list of arrays
    - `apply` which updates the first p.Array instance

    Mutation should take only one p.Array instance as argument, while
    Recombinations should take several
    """

    def __init__(self, **parameters: tp.Any) -> None:
        self._parameters = parameters
        super().__init__()

    def root(self) -> Data:
        data = self._layers[0]
        if not isinstance(data, Data):
            raise RuntimeError(
                f"{self.__class__.__name__} must be applied to Data parameters, got: {type(data)}"
            )
        return data

    def _on_layer_added(self) -> None:
        params = self.root().parameters
        for name, obj in self._parameters.items():
            if name not in params:
                params[name] = core.as_parameter(obj)
        self._parameters = {}

    def __call__(self, data: D, inplace: bool = False) -> D:
        new = data if inplace else data.copy()
        new.add_layer(self.copy())
        return new


class Crossover(Mutation):
    """Operator for merging part of an array into another one

    Parameters
    ----------
    axis: None or int or tuple of ints
        the axis (or axes) on which the merge will happen. This axis will be split into 3 parts: the first and last one will take
        value from the first array, the middle one from the second array.
    max_size: None or int
        maximum size of the part taken from the second array. By default, this is at most around half the number of total elements of the
        array to the power of 1/number of axis.


    Notes
    -----
    - this is experimental, the API may evolve
    - when using several axis, the size of the second array part is the same on each axis (aka a square in 2D, a cube in 3D, ...)

    Examples:
    ---------
    - 2-dimensional array, with crossover on dimension 1:
      0 1 0 0
      0 1 0 0
      0 1 0 0
    - 2-dimensional array, with crossover on dimensions 0 and 1:
      0 0 0 0
      0 1 1 0
      0 1 1 0
    """

    def __init__(
        self,
        axis: tp.Optional[tp.Union[int, tp.Iterable[int]]] = None,
        max_size: tp.Optional[int] = None,
        fft: bool = False,
    ) -> None:
        if not isinstance(axis, core.Parameter):
            axis = (axis,) if isinstance(axis, int) else tuple(axis) if axis is not None else None
        super().__init__(max_size=max_size, axis=axis, fft=fft)

    @property
    def axis(self) -> tp.Optional[tp.Tuple[int, ...]]:
        return self.root().parameters["axis"].value  # type: ignore

    def _layered_recombine(self, *arrays: Data) -> None:  # type: ignore
        root = self.root()
        new_value = self._apply_array([root.value] + [a.value for a in arrays])
        bounds = root.bounds
        if root.parameters["fft"].value and any(x is not None for x in bounds):
            new_value = transforms.Clipping(a_min=bounds[0], a_max=bounds[1]).forward(new_value)
        root.value = new_value

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        root = self.root()
        # checks
        if len(arrays) != 2:
            raise Exception(f"Crossover can only be applied between 2 individuals, got {len(arrays)}")
        transf = (
            transforms.Fourrier(range(arrays[0].dim) if self.axis is None else self.axis)
            if root.parameters["fft"].value
            else None
        )
        if transf is not None:
            arrays = [transf.forward(a) for a in arrays]
        shape = arrays[0].shape
        assert shape == arrays[1].shape, "Individuals should have the same shape"
        # settings
        axis = tuple(range(len(shape))) if self.axis is None else self.axis
        max_size = root.parameters["max_size"].value
        max_size = int(((arrays[0].size + 1) / 2) ** (1 / len(axis))) if max_size is None else max_size
        max_size = min(max_size, *(shape[a] - 1 for a in axis))
        size = 1 if max_size == 1 else self.random_state.randint(1, max_size)
        # slices
        slices = _make_slices(shape, axis, size, self.random_state)
        result = np.array(arrays[0], copy=True)
        result[tuple(slices)] = arrays[1][tuple(slices)]
        if transf is not None:
            result = transf.backward(result)
        return result


class RavelCrossover(Crossover):
    """Operator for merging part of an array into another one, after raveling

    Parameters
    ----------
    max_size: None or int
        maximum size of the part taken from the second array. By default, this is at most around half the number of total elements of the
        array to the power of 1/number of axis.
    """

    def __init__(
        self,
        max_size: tp.Optional[int] = None,
    ) -> None:
        super().__init__(axis=0, max_size=max_size)

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        shape = arrays[0].shape
        out = super()._apply_array([a.ravel() for a in arrays])
        return out.reshape(shape)


def _make_slices(
    shape: tp.Tuple[int, ...], axes: tp.Tuple[int, ...], size: int, rng: np.random.RandomState
) -> tp.List[slice]:
    slices = []
    for a, s in enumerate(shape):
        if a in axes:
            if s <= 1:
                raise ValueError("Cannot crossover on axis with size 1")
            start = rng.randint(s - size)
            slices.append(slice(start, start + size))
        else:
            slices.append(slice(None))
    return slices


class Translation(Mutation):
    def __init__(self, axis: tp.Optional[tp.Union[int, tp.Iterable[int]]] = None):
        if not isinstance(axis, core.Parameter):
            axis = (axis,) if isinstance(axis, int) else tuple(axis) if axis is not None else None
        super().__init__(axis=axis)

    @property
    def axis(self) -> tp.Optional[tp.Tuple[int, ...]]:
        return self.root().parameters["axis"].value  # type: ignore

    def _layered_mutate(self) -> None:
        root = self.root()
        root._value = self._apply_array([root._value])

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        assert len(arrays) == 1
        data = arrays[0]
        axis = tuple(range(data.dim)) if self.axis is None else self.axis
        shifts = [self.random_state.randint(data.shape[a]) for a in axis]
        return np.roll(data, shifts, axis=axis)  # type: ignore


class AxisSlicedArray:
    def __init__(self, array: np.ndarray, axis: int):
        self.array = array
        self.axis = axis

    def __getitem__(self, slice_: slice) -> np.ndarray:
        assert isinstance(slice_, slice)
        slices = tuple(slice_ if a == self.axis else slice(None) for a in range(self.array.ndim))
        return self.array[slices]  # type: ignore


class Jumping(Mutation):
    """Move a chunk for a position to another in an array"""

    def __init__(self, axis: int, size: int):
        super().__init__(axis=axis, size=size)

    @property
    def axis(self) -> int:
        return self.root().parameters["axis"].value  # type: ignore

    @property
    def size(self) -> int:
        return self.root().parameters["size"].value  # type: ignore

    def _layered_mutate(self) -> None:
        root = self.root()
        root._value = self._apply_array([root._value])

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        assert len(arrays) == 1
        data = arrays[0]
        L = data.shape[self.axis]
        size = self.random_state.randint(1, self.size)
        asdata = AxisSlicedArray(data, self.axis)
        init = self.random_state.randint(L)
        chunck = asdata[init : init + size]
        remain: np.ndarray = np.concatenate([asdata[:init], asdata[init + size :]], axis=self.axis)
        # pylint: disable=unsubscriptable-object
        newpos = self.random_state.randint(remain.shape[self.axis])
        asremain = AxisSlicedArray(remain, self.axis)
        return np.concatenate([asremain[:newpos], chunck, asremain[newpos:]], axis=self.axis)  # type: ignore


class LocalGaussian(Mutation):
    def __init__(
        self, size: tp.Union[int, core.Parameter], axes: tp.Optional[tp.Union[int, tp.Iterable[int]]] = None
    ):
        if not isinstance(axes, core.Parameter):
            axes = (axes,) if isinstance(axes, int) else tuple(axes) if axes is not None else None
        super().__init__(axes=axes, size=size)

    @property
    def axes(self) -> tp.Optional[tp.Tuple[int, ...]]:
        return self.root().parameters["axes"].value  # type: ignore

    def _layered_mutate(self) -> None:
        root = self.root()
        data = np.zeros(root.value.shape)
        # settings
        axis = tuple(range(len(data.shape))) if self.axes is None else self.axes
        size = self.root().parameters["size"].value
        # slices
        slices = _make_slices(data.shape, axis, size, self.random_state)
        shape = data[tuple(slices)].shape
        data[tuple(slices)] += self.random_state.normal(0, 1, size=shape)
        root._internal_set_standardized_data(data.ravel(), reference=root)


def rolling_mean(vector: np.ndarray, window: int) -> np.ndarray:
    if window >= len(vector):
        return np.sum(vector) * np.ones((len(vector),))  # type: ignore
    if window <= 1:
        return vector
    cumsum: np.ndarray = np.cumsum(np.concatenate(([0], vector, vector[: window - 1])))
    return cumsum[window:] - cumsum[:-window]  # type: ignore


# class TunedTranslation(Mutation):
#     def __init__(self, axis: int, shape: tp.Sequence[int]):
#         assert isinstance(axis, int)
#         self.shape = tuple(shape)
#         super().__init__(shift=Choice(range(1, shape[axis])))
#         self.axis = axis
#
#     @property
#     def shift(self) -> Choice:
#         return self.root().parameters["shift"]  # type: ignore
#
#     def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
#         assert len(arrays) == 1
#         data = arrays[0]
#         assert data.shape == self.shape
#         shift = self.shift.value
#         # update shift arrray
#         shifts = self.shift.indices._value
#         self.shift.indices._value = np.roll(shifts, shift)  # update probas
#         return np.roll(data, shift, axis=self.axis)  # type: ignore


# class ProbaLocalGaussian(Mutation):
#     def __init__(self, axis: int, shape: tp.Sequence[int]):
#         assert isinstance(axis, int)
#         self.shape = tuple(shape)
#         self.axis = axis
#         super().__init__(
#             positions=Array(shape=(shape[axis],)),
#             ratio=Scalar(init=1, lower=0, upper=1).set_mutation(sigma=0.05),
#         )
#
#     def axes(self) -> tp.Optional[tp.Tuple[int, ...]]:
#         return self.root().parameters["axes"].value  # type: ignore
#
#     def apply(self, arrays: tp.Sequence[Data]) -> None:
#         arrays = list(arrays)
#         assert len(arrays) == 1
#         data = np.zeros(arrays[0].value.shape)
#         # settings
#         length = self.shape[self.axis]
#         size = int(max(1, np.round(length * self.root().parameters["ratio"].value)))
#         # slices
#         e_weights = np.exp(rolling_mean(self.root().parameters["positions"].value, size))
#         probas = e_weights / np.sum(e_weights)
#         index = self.random_state.choice(range(length), p=probas)
#         # update (inefficient)
#         shape = tuple(size if a == self.axis else s for a, s in enumerate(arrays[0].value.shape))
#         data[tuple(slice(s) for s in shape)] += self.random_state.normal(0, 1, size=shape)
#         data = np.roll(data, shift=index, axis=self.axis)
#         arrays[0]._internal_set_standardized_data(data.ravel(), reference=arrays[0])
