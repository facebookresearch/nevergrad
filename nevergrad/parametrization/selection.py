# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
from ..instrumentation import discretization  # TODO move along
from .core import Parameter
from .core import _as_parameter
from .core import Dict
from .core import Tags
from .container import Tuple
from .data import Array


C = t.TypeVar("C", bound="Choice")


# TODO deterministic in name + Ordered + ordered tag
class Choice(Dict):
    """Parameter which choses one of the provided choice options as a value.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn randomly from the softmax of weights which are
    updated during the optimization.

    Parameters
    ----------
    choices: list
        a list of possible values or Parameters for the variable.
    deterministic: bool
        whether to always draw the most likely choice (hence avoiding the stochastic behavior, but loosing
        continuity)

    Note
    ----
    - Since the chosen value is drawn randomly, the use of this variable makes deterministic
      functions become stochastic, hence "adding noise"
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched
    """

    def __init__(
            self,
            choices: t.Iterable[t.Any],
            deterministic: bool = False,
    ) -> None:
        assert not isinstance(choices, Tuple)
        lchoices = list(choices)  # for iterables
        super().__init__(weights=Array(shape=(len(lchoices),), mutable_sigma=False),
                         choices=Tuple(*lchoices))
        self._deterministic = deterministic
        self._index: t.Optional[int] = None

    @property
    def tags(self) -> Tags:
        return Tags(deterministic=self._deterministic & self.choices.tags.deterministic,
                    continuous=self.choices.tags.continuous & (not self._deterministic))

    @property
    def index(self) -> int:  # delayed choice
        """Index of the chosen option
        """
        if self._index is None:
            self._draw(deterministic=self._deterministic)
        assert self._index is not None
        return self._index

    @property
    def weights(self) -> Array:
        """The weights used to draw the value
        """
        return self["weights"]  # type: ignore

    @property
    def choices(self) -> Tuple:
        """The different options, as a Tuple Parameter
        """
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
        self.weights.set_std_data(out, deterministic=True)
        self._index = index

    def get_value_hash(self) -> t.Hashable:
        return (self.index, _as_parameter(self.choices[self.index]).get_value_hash())

    def _draw(self, deterministic: bool = True) -> None:
        weights = self.weights.value
        random = False if deterministic or self._deterministic else self.random_state
        self._index = int(discretization.softmax_discretization(weights, weights.size, random=random)[0])

    def _internal_set_std_data(self: C, data: np.ndarray, instance: C, deterministic: bool = False) -> C:
        super()._internal_set_std_data(data, instance=instance, deterministic=deterministic)
        instance._draw(deterministic=deterministic)
        return instance

    def mutate(self) -> None:
        self.weights.mutate()
        self._draw(deterministic=self._deterministic)
        param = self.choices[self.index]
        if isinstance(param, Parameter):
            param.mutate()

    def _internal_spawn_child(self: C) -> C:
        child = self.__class__(choices=[], deterministic=self._deterministic)
        child._parameters["choices"] = self.choices.spawn_child()
        child._parameters["weights"] = self.weights.spawn_child()
        return child
