# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
from . import discretization
from .core3 import Parameter
from .core3 import _as_parameter
from .container import NgDict
from .container import NgTuple
from .data import Array


C = t.TypeVar("C", bound="Choice")


class Choice(NgDict):

    def __init__(
            self,
            choices: t.Iterable[t.Any],
            recombination: t.Union[str, Parameter] = "average",
            deterministic: bool = False,
    ) -> None:
        assert not isinstance(choices, NgTuple)
        lchoices = list(choices)  # for iterables
        super().__init__(probabilities=Array(shape=(len(lchoices),), recombination=recombination),
                         choices=NgTuple(*lchoices))
        self._deterministic = deterministic
        self._index: t.Optional[int] = None

    @property
    def index(self) -> int:  # delayed choice
        if self._index is None:
            self._draw(deterministic=self._deterministic)
        assert self._index is not None
        return self._index

    @property
    def probabilities(self) -> Array:
        return self["probabilities"]  # type: ignore

    @property
    def choices(self) -> NgTuple:
        return self["choices"]  # type: ignore

    @property
    def value(self) -> t.Any:
        return _as_parameter(self.choices[self.index]).value

    @value.setter
    def value(self, value: t.Any) -> None:
        index = -1
        # try to find where to put this
        nums = sorted(int(k) for k in self.choices._parameters)
        for k in nums:
            choice = _as_parameter(self.choices[k])
            try:
                choice.value = value
            except Exception:  # pylint: disable=broad-except
                pass
            else:
                index = int(k)
                break
        if index == -1:
            raise ValueError(f"Could not figure out where to put value {value}")
        out = discretization.inverse_softmax_discretization(index, len(nums))
        self.probabilities.set_std_data(out, deterministic=True)
        self._index = index

    def get_value_hash(self) -> t.Hashable:
        return (self.index, _as_parameter(self.choices[self.index]).get_value_hash())

    def _draw(self, deterministic: bool = True) -> None:
        probas = self.probabilities.value
        random = False if deterministic or self._deterministic else self.random_state
        self._index = int(discretization.softmax_discretization(probas, probas.size, random=random)[0])

    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        super().set_std_data(data, deterministic=deterministic)
        self._draw(deterministic=deterministic)

    def mutate(self) -> None:
        self.probabilities.mutate()
        self._draw(deterministic=self._deterministic)
        param = self.choices[self.index]
        if isinstance(param, Parameter):
            param.mutate()

    def _internal_spawn_child(self: C) -> C:
        child = self.__class__(choices=[], deterministic=self._deterministic)
        child._parameters["choices"] = self.choices.spawn_child()
        child._parameters["probabilities"] = self.probabilities.spawn_child()
        return child
