# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
from nevergrad.common.typetools import ArrayLike
from ..instrumentation import discretization  # TODO move along
from . import core
from .container import Tuple
from .data import Array
# weird pylint issue on "Descriptors"
# pylint: disable=no-value-for-parameter


C = t.TypeVar("C", bound="Choice")
T = t.TypeVar("T", bound="TransitionChoice")


class BaseChoice(core.Dict):

    def __init__(self, *, choices: t.Iterable[t.Any], **kwargs: t.Any) -> None:
        assert not isinstance(choices, Tuple)
        lchoices = list(choices)  # for iterables
        super().__init__(choices=Tuple(*lchoices), **kwargs)
        self._index: t.Optional[int] = None

    @property
    def descriptors(self) -> core.Descriptors:
        return core.Descriptors(deterministic=self.choices.descriptors.deterministic,
                                continuous=self.choices.descriptors.continuous)

    @property
    def index(self) -> int:
        assert self._index is not None
        return self._index

    @property
    def choices(self) -> Tuple:
        """The different options, as a Tuple Parameter
        """
        return self["choices"]  # type: ignore

    @property
    def value(self) -> t.Any:
        return core.as_parameter(self.choices[self.index]).value

    @value.setter
    def value(self, value: t.Any) -> None:
        self._find_and_set_value(value)

    def _find_and_set_value(self, value: t.Any) -> None:
        index = -1
        # try to find where to put this
        nums = sorted(int(k) for k in self.choices._parameters)
        for k in nums:
            choice = core.as_parameter(self.choices[k])
            try:
                choice.value = value
            except Exception:  # pylint: disable=broad-except
                pass
            else:
                index = int(k)
                break
        if index == -1:
            raise ValueError(f"Could not figure out where to put value {value}")
        self._index = index

    def get_value_hash(self) -> t.Hashable:
        return (self.index, core.as_parameter(self.choices[self.index]).get_value_hash())


# TODO ordered tag
class Choice(BaseChoice):
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
        lchoices = list(choices)
        super().__init__(choices=lchoices, weights=Array(shape=(len(lchoices),), mutable_sigma=False))
        self._deterministic = deterministic

    def _get_name(self) -> str:
        name = super()._get_name()
        cls = self.__class__.__name__
        assert name.startswith(cls)
        if self._deterministic:
            name = cls + "{det}" + name[len(cls):]
        return name

    @property
    def descriptors(self) -> core.Descriptors:
        return core.Descriptors(deterministic=self._deterministic & self.choices.descriptors.deterministic,
                                continuous=self.choices.descriptors.continuous & (not self._deterministic))

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

    def _find_and_set_value(self, value: t.Any) -> None:
        super()._find_and_set_value(value)
        # force new probabilities
        out = discretization.inverse_softmax_discretization(self.index, len(self))
        self.weights.set_std_data(out, deterministic=True)

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
        if isinstance(param, core.Parameter):
            param.mutate()

    def _internal_spawn_child(self: C) -> C:
        child = self.__class__(choices=[], deterministic=self._deterministic)
        child._parameters["choices"] = self.choices.spawn_child()
        child._parameters["weights"] = self.weights.spawn_child()
        return child


class TransitionChoice(BaseChoice):
    """Parameter which choses one of the provided choice options as a value.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn using transitions between current choice and the next/previous ones.

    Parameters
    ----------
    choices: list
        a list of possible values or Parameters for the variable.
    transitions: np.ndarray or Array
        the transition weights. During transition, the direction (forward or backward will be drawn with
        equal probabilities), then the transitions weights are normalized through softmax, the 1st value gives
        the probability to remain in the same state, the second to move one step (backward or forward) and so on.

    Note
    ----
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched
    """

    def __init__(
            self,
            choices: t.Iterable[t.Any],
            transitions: t.Union[ArrayLike, Array] = (1.0, 1.0),
    ) -> None:
        super().__init__(choices=choices, transitions=transitions if isinstance(transitions, Array) else np.array(transitions, copy=False))
        assert core.as_parameter(self.transitions).value.ndim == 1
        self._index = (len(self.choices) - 1) // 2  # middle or just below

    @property
    def transitions(self) -> t.Union[np.ndarray, Array]:
        """The weights used to draw the value
        """
        return self["transitions"]  # type: ignore

    def mutate(self) -> None:
        transitions = core.as_parameter(self.transitions)
        transitions.mutate()
        probas = np.exp(transitions.value)
        probas /= np.sum(probas)
        move = self.random_state.choice(list(range(probas.size)), p=probas)
        sign = 1 if self.random_state.randint(2) else -1
        self._index = max(0, min(len(self.choices), self._index + sign * move))
        # mutate corresponding parameter
        param = self.choices[self.index]
        if isinstance(param, core.Parameter):
            param.mutate()

    def _internal_spawn_child(self: T) -> T:
        child = self.__class__(choices=[])
        child._parameters["choices"] = self.choices.spawn_child()
        child._parameters["transitions"] = (np.array(self.transitions) if not isinstance(self.transitions, Array)
                                            else self.transitions.spawn_child())
        child._index = self._index
        return child
