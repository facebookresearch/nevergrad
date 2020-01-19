# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.common.typetools import ArrayLike
from . import discretization
from . import utils
from . import core
from .container import Tuple
from .data import Array
from .data import Scalar
# weird pylint issue on "Descriptors"
# pylint: disable=no-value-for-parameter


C = tp.TypeVar("C", bound="Choice")
T = tp.TypeVar("T", bound="TransitionChoice")


class BaseChoice(core.Dict):

    def __init__(self, *, choices: tp.Iterable[tp.Any], **kwargs: tp.Any) -> None:
        assert not isinstance(choices, Tuple)
        lchoices = list(choices)  # for iterables
        super().__init__(choices=Tuple(*lchoices), **kwargs)

    def _compute_descriptors(self) -> utils.Descriptors:
        deterministic = getattr(self, "_deterministic", True)
        ordered = not hasattr(self, "_deterministic")
        internal = utils.Descriptors(deterministic=deterministic, continuous=not deterministic, ordered=ordered)
        return self.choices.descriptors & internal

    def __len__(self) -> int:
        """Number of choices
        """
        return len(self.choices)

    @property
    def index(self) -> int:
        raise Exception

    @property
    def choices(self) -> Tuple:
        """The different options, as a Tuple Parameter
        """
        return self["choices"]  # type: ignore

    @property
    def value(self) -> tp.Any:
        return core.as_parameter(self.choices[self.index]).value

    @value.setter
    def value(self, value: tp.Any) -> None:
        self._find_and_set_value(value)

    def _find_and_set_value(self, value: tp.Any) -> int:
        self._check_frozen()
        index = -1
        # try to find where to put this
        nums = sorted(int(k) for k in self.choices._content)
        for k in nums:
            choice = self.choices[k]
            try:
                choice.value = value
                index = k
                break
            except Exception:  # pylint: disable=broad-except
                pass
        if index == -1:
            raise ValueError(f"Could not figure out where to put value {value}")
        return index

    def get_value_hash(self) -> tp.Hashable:
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
            choices: tp.Iterable[tp.Any],
            deterministic: bool = False,
    ) -> None:
        assert not isinstance(choices, Tuple)
        lchoices = list(choices)
        super().__init__(choices=lchoices, weights=Array(shape=(len(lchoices),), mutable_sigma=False))
        self._deterministic = deterministic
        self._index: tp.Optional[int] = None

    def _get_name(self) -> str:
        name = super()._get_name()
        cls = self.__class__.__name__
        assert name.startswith(cls)
        if self._deterministic:
            name = cls + "{det}" + name[len(cls):]
        return name

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

    def _find_and_set_value(self, value: tp.Any) -> int:
        index = super()._find_and_set_value(value)
        self._index = index
        # force new probabilities
        out = discretization.inverse_softmax_discretization(self.index, len(self))
        self.weights._value *= 0.  # reset since there is no reference
        self.weights.set_standardized_data(out, deterministic=True)
        return index

    def _draw(self, deterministic: bool = True) -> None:
        weights = self.weights.value
        random = False if deterministic or self._deterministic else self.random_state
        self._index = int(discretization.softmax_discretization(weights, weights.size, random=random)[0])

    def _internal_set_standardized_data(self: C, data: np.ndarray, reference: C, deterministic: bool = False) -> None:
        super()._internal_set_standardized_data(data, reference=reference, deterministic=deterministic)
        self._draw(deterministic=deterministic)

    def mutate(self) -> None:
        self.weights.mutate()
        self._draw(deterministic=self._deterministic)
        self.choices[self.index].mutate()

    def _internal_spawn_child(self: C) -> C:
        child = self.__class__(choices=[], deterministic=self._deterministic)
        child._content["choices"] = self.choices.spawn_child()
        child._content["weights"] = self.weights.spawn_child()
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
    - in order to support export to standardized space, the index is encoded as a scalar. A normal distribution N(O,1)
      on this scalar yields a uniform choice of index. This may come to evolve for simplicity's sake.
    - currently, transitions are computed through softmax, this may evolve since this is somehow impractical
    """

    def __init__(
            self,
            choices: tp.Iterable[tp.Any],
            transitions: tp.Union[ArrayLike, Array] = (1.0, 1.0),
    ) -> None:
        super().__init__(choices=choices,
                         position=Scalar(),
                         transitions=transitions if isinstance(transitions, Array) else np.array(transitions, copy=False))
        assert self.transitions.value.ndim == 1

    @property
    def index(self) -> int:
        return discretization.threshold_discretization(np.array([self.position.value]), arity=len(self.choices))[0]

    def _find_and_set_value(self, value: tp.Any) -> int:
        index = super()._find_and_set_value(value)
        self._set_index(index)
        return index

    def _set_index(self, index: int) -> None:
        out = discretization.inverse_threshold_discretization([index], len(self.choices))
        self.position.value = out[0]

    @property
    def transitions(self) -> Array:
        """The weights used to draw the step to the next value
        """
        return self["transitions"]  # type: ignore

    @property
    def position(self) -> Scalar:
        """The continuous version of the index (used when working with standardized space)
        """
        return self["position"]  # type: ignore

    def mutate(self) -> None:
        transitions = core.as_parameter(self.transitions)
        transitions.mutate()
        probas = np.exp(transitions.value)
        probas /= np.sum(probas)  # TODO decide if softmax is the best way to go...
        move = self.random_state.choice(list(range(probas.size)), p=probas)
        sign = 1 if self.random_state.randint(2) else -1
        new_index = max(0, min(len(self.choices), self.index + sign * move))
        self._set_index(new_index)
        # mutate corresponding parameter
        self.choices[self.index].mutate()

    def _internal_spawn_child(self: T) -> T:
        child = self.__class__(choices=[])
        child._content["choices"] = self.choices.spawn_child()
        child._content["position"] = self.position.spawn_child()
        child._content["transitions"] = self.transitions.spawn_child()
        return child
