# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from . import core
from .data import Mutation
from .choice import Choice


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

    def __init__(self, axis: tp.Optional[tp.Union[int, tp.Iterable[int]]] = None, max_size: tp.Optional[int] = None) -> None:
        if not isinstance(axis, core.Parameter):
            axis = (axis,) if isinstance(axis, int) else tuple(axis) if axis is not None else None
        super().__init__(max_size=max_size, axis=axis)

    @property
    def axis(self) -> tp.Optional[tp.Tuple[int, ...]]:
        return self.parameters["axis"].value  # type: ignore

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        # checks
        arrays = list(arrays)
        if len(arrays) != 2:
            raise Exception("Crossover can only be applied between 2 individuals")
        shape = arrays[0].shape
        assert shape == arrays[1].shape, "Individuals should have the same shape"
        # settings
        axis = tuple(range(len(shape))) if self.axis is None else self.axis
        max_size = self.parameters["max_size"].value
        max_size = int(((arrays[0].size + 1) / 2)**(1 / len(axis))) if max_size is None else max_size
        max_size = min(max_size, *(shape[a] - 1 for a in axis))
        size = 1 if max_size == 1 else self.random_state.randint(1, max_size)
        # slices
        slices = []
        for a, s in enumerate(shape):
            if a in axis:
                if s <= 1:
                    raise ValueError("Cannot crossover an shape with size 1")
                start = self.random_state.randint(s - size)
                slices.append(slice(start, start + size))
            else:
                slices.append(slice(0, s))
        result = np.array(arrays[0], copy=True)
        result[tuple(slices)] = arrays[1][tuple(slices)]
        return result


class Translation(Mutation):

    def __init__(self, axis: tp.Optional[tp.Union[int, tp.Iterable[int]]]):
        if not isinstance(axis, core.Parameter):
            axis = (axis,) if isinstance(axis, int) else tuple(axis) if axis is not None else None
        super().__init__(axis=axis)

    @property
    def axis(self) -> tp.Optional[tp.Tuple[int, ...]]:
        return self.parameters["axis"].value  # type: ignore

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        arrays = list(arrays)
        assert len(arrays) == 1
        data = arrays[0]
        axis = tuple(range(data.dim)) if self.axis is None else self.axis
        shifts = [self.random_state.randint(data.shape[a]) for a in axis]
        return np.roll(data, shifts, axis=axis)  # type: ignore


class TunedTranslation(Mutation):

    def __init__(self, axis: int, shape: tp.Sequence[int]):
        assert isinstance(axis, int)
        self.shape = tuple(shape)
        super().__init__(shift=Choice(range(1, shape[axis])))
        self.axis = axis

    @property
    def shift(self) -> Choice:
        return self.parameters["shift"]  # type: ignore

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        arrays = list(arrays)
        assert len(arrays) == 1
        data = arrays[0]
        assert data.shape == self.shape
        shift = self.shift.value
        # update shift arrray
        shifts = self.shift.weights.value
        self.shift.weights.value = np.roll(shifts, shift)  # update probas
        return np.roll(data, shift, axis=self.axis)  # type: ignore

    def _internal_spawn_child(self) -> "TunedTranslation":
        child = self.__class__(axis=self.axis, shape=self.shape)
        child.parameters._content = {k: v.spawn_child() if isinstance(v, core.Parameter) else v
                                     for k, v in self.parameters._content.items()}
        return child
