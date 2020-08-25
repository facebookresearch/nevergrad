# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import nevergrad.common.typing as tp
from . import discretization
from . import utils
from . import core
from .container import Tuple
from .data import Array
# weird pylint issue on "Descriptors"
# pylint: disable=no-value-for-parameter


C = tp.TypeVar("C", bound="Choice")
T = tp.TypeVar("T", bound="TransitionChoice")


class BaseChoice(core.Dict):

    def __init__(
        self,
        *,
        choices: tp.Iterable[tp.Any],
        repetitions: tp.Optional[int] = None,
        **kwargs: tp.Any
    ) -> None:
        assert repetitions is None or isinstance(repetitions, int)  # avoid silent issues
        self._repetitions = repetitions
        assert not isinstance(choices, Tuple)
        lchoices = list(choices)  # for iterables
        if not lchoices:
            raise ValueError("{self._class__.__name__} received an empty list of options.")
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
    def index(self) -> int:  # delayed choice
        """Index of the chosen option
        """
        assert self.indices.size == 1
        return int(self.indices[0])

    @property
    def indices(self) -> np.ndarray:
        """Indices of the chosen options
        """
        raise NotImplementedError  # TODO remove index?

    @property
    def choices(self) -> Tuple:
        """The different options, as a Tuple Parameter
        """
        return self["choices"]  # type: ignore

    @property
    def value(self) -> tp.Any:
        return self._get_value()

    @value.setter
    def value(self, value: tp.Any) -> None:
        self._find_and_set_value(value)

    def _get_value(self) -> tp.Any:
        if self._repetitions is None:
            return core.as_parameter(self.choices[self.index]).value
        return tuple(core.as_parameter(self.choices[ind]).value for ind in self.indices)

    def _find_and_set_value(self, values: tp.List[tp.Any]) -> np.ndarray:
        """Must be adapted to each class
        This handles a list of values, not just one
        """  # TODO this is currenlty very messy, may need some improvement
        values = [values] if self._repetitions is None else values
        self._check_frozen()
        indices: np.ndarray = -1 * np.ones(len(values), dtype=int)
        nums = sorted(int(k) for k in self.choices._content)
        # try to find where to put this
        for i, value in enumerate(values):
            for k in nums:
                choice = self.choices[k]
                try:
                    choice.value = value
                    indices[i] = k
                    break
                except Exception:  # pylint: disable=broad-except
                    pass
            if indices[i] == -1:
                raise ValueError(f"Could not figure out where to put value {value}")
        return indices

    def get_value_hash(self) -> tp.Hashable:
        hashes: tp.List[tp.Hashable] = []
        for ind in self.indices:
            c = self.choices[int(ind)]
            const = isinstance(c, core.Constant) or not isinstance(c, core.Parameter)
            hashes.append(int(ind) if const else (int(ind), c.get_value_hash()))
        return tuple(hashes) if len(hashes) > 1 else hashes[0]


class Choice(BaseChoice):
    """Unordered categorical parameter, randomly choosing one of the provided choice options as a value.
    The choices can be Parameters, in which case there value will be returned instead.
    The chosen parameter is drawn randomly from the softmax of weights which are
    updated during the optimization.

    Parameters
    ----------
    choices: list
        a list of possible values or Parameters for the variable.
    repetitions: None or int
        set to an integer :code:`n` if you want :code:`n` similar choices sampled independently (each with its own distribution)
        This is equivalent to :code:`Tuple(*[Choice(options) for _ in range(n)])` but can be
        30x faster for large :code:`n`.
    deterministic: bool
        whether to always draw the most likely choice (hence avoiding the stochastic behavior, but loosing
        continuity)

    Note
    ----
    - Since the chosen value is drawn randomly, the use of this variable makes deterministic
      functions become stochastic, hence "adding noise"
    - the "mutate" method only mutates the weights and the chosen Parameter (if it is not constant),
      leaving others untouched

    Examples
    --------

    >>> print(Choice(["a", "b", "c", "e"]).value)
    "c"

    >>> print(Choice(["a", "b", "c", "e"], repetitions=3).value)
    ("b", "b", "c")
    """

    def __init__(
            self,
            choices: tp.Iterable[tp.Any],
            repetitions: tp.Optional[int] = None,
            deterministic: bool = False,
    ) -> None:
        assert not isinstance(choices, Tuple)
        lchoices = list(choices)
        rep = 1 if repetitions is None else repetitions
        super().__init__(choices=lchoices, repetitions=repetitions,
                         weights=Array(shape=(rep, len(lchoices)), mutable_sigma=False))
        self._deterministic = deterministic
        self._indices: tp.Optional[np.ndarray] = None

    def _get_name(self) -> str:
        name = super()._get_name()
        cls = self.__class__.__name__
        assert name.startswith(cls)
        if self._deterministic:
            name = cls + "{det}" + name[len(cls):]
        return name

    @property
    def indices(self) -> np.ndarray:  # delayed choice
        """Index of the chosen option
        """
        if self._indices is None:
            self._draw(deterministic=self._deterministic)
        assert self._indices is not None
        return self._indices

    @property
    def weights(self) -> Array:
        """The weights used to draw the value
        """
        return self["weights"]  # type: ignore

    @property
    def probabilities(self) -> np.ndarray:
        """The probabilities used to draw the value
        """
        exp = np.exp(self.weights.value)
        return exp / np.sum(exp)

    def _find_and_set_value(self, values: tp.Any) -> np.ndarray:
        indices = super()._find_and_set_value(values)
        self._indices = indices
        # force new probabilities
        arity = self.weights.value.shape[1]
        coeff = discretization.weight_for_reset(arity)
        self.weights._value.fill(0.0)  # reset since there is no reference
        out = np.array(self.weights._value, copy=True)  # just a zero matrix
        out[np.arange(indices.size), indices] = coeff
        self.weights.set_standardized_data(out.ravel(), deterministic=True)
        return indices

    def _draw(self, deterministic: bool = True) -> None:
        encoder = discretization.Encoder(self.weights.value, rng=self.random_state)
        self._indices = encoder.encode(deterministic=deterministic or self._deterministic)

    def _internal_set_standardized_data(self: C, data: np.ndarray, reference: C, deterministic: bool = False) -> None:
        super()._internal_set_standardized_data(data, reference=reference, deterministic=deterministic)
        self._draw(deterministic=deterministic)

    def mutate(self) -> None:
        # force random_state sync
        self.random_state   # pylint: disable=pointless-statement
        self.weights.mutate()
        self._draw(deterministic=self._deterministic)
        indices = set(self.indices)
        for ind in indices:
            self.choices[ind].mutate()

    def _internal_spawn_child(self: C) -> C:
        choices = (y for x, y in sorted(self.choices.spawn_child()._content.items()))
        child = self.__class__(choices=choices, deterministic=self._deterministic, repetitions=self._repetitions)
        child._content["weights"] = self.weights.spawn_child()
        return child


class TransitionChoice(BaseChoice):
    """Ordered categorical parameter, choosing one of the provided choice options as a value, with continuous transitions.
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
            transitions: tp.Union[tp.ArrayLike, Array] = (1.0, 1.0),
            repetitions: tp.Optional[int] = None,
    ) -> None:
        choices = list(choices)
        positions = Array(init=len(choices) / 2.0 * np.ones((repetitions if repetitions is not None else 1,)))
        positions.set_bounds(0, len(choices), method="gaussian")
        super().__init__(choices=choices,
                         repetitions=repetitions,
                         positions=positions,
                         transitions=transitions if isinstance(transitions, Array) else np.array(transitions, copy=False))
        assert self.transitions.value.ndim == 1

    @property
    def indices(self) -> np.ndarray:
        return np.minimum(len(self) - 1e-9, self.positions.value).astype(int)

    def _find_and_set_value(self, values: tp.Any) -> np.ndarray:
        indices = super()._find_and_set_value(values)  # only one value for this class
        self._set_index(indices)
        return indices

    def _set_index(self, indices: np.ndarray) -> None:
        self.positions.value = indices + 0.5

    @property
    def transitions(self) -> Array:
        """The weights used to draw the step to the next value
        """
        return self["transitions"]  # type: ignore

    @property
    def position(self) -> Array:
        """The continuous version of the index (used when working with standardized space)
        """
        warnings.warn("position is replaced by positions in order to allow for repetitions", DeprecationWarning)
        return self.positions

    @property
    def positions(self) -> Array:
        """The continuous version of the index (used when working with standardized space)
        """
        return self["positions"]  # type: ignore

    def mutate(self) -> None:
        # force random_state sync
        self.random_state   # pylint: disable=pointless-statement
        transitions = core.as_parameter(self.transitions)
        transitions.mutate()
        rep = 1 if self._repetitions is None else self._repetitions
        #
        enc = discretization.Encoder(np.ones((rep, 1)) * np.log(self.transitions.value),
                                     self.random_state)
        moves = enc.encode()
        signs = self.random_state.choice([-1, 1], size=rep)
        new_index = np.clip(self.indices + signs * moves, 0, len(self) - 1)
        self._set_index(new_index.ravel())
        # mutate corresponding parameter
        indices = set(self.indices)
        for ind in indices:
            self.choices[ind].mutate()

    def _internal_spawn_child(self: T) -> T:
        choices = (y for x, y in sorted(self.choices.spawn_child()._content.items()))
        child = self.__class__(choices=choices, repetitions=self._repetitions)
        child._content["positions"] = self.positions.spawn_child()
        child._content["transitions"] = self.transitions.spawn_child()
        return child
