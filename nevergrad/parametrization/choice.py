# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import nevergrad.common.typing as tp
from . import discretization
from . import core
from . import container
from . import _datalayers
from .data import Array

# weird pylint issue on "Descriptors"
# pylint: disable=no-value-for-parameter


C = tp.TypeVar("C", bound="Choice")
T = tp.TypeVar("T", bound="TransitionChoice")


class BaseChoice(container.Container):
    def __init__(
        self, *, choices: tp.Iterable[tp.Any], repetitions: tp.Optional[int] = None, **kwargs: tp.Any
    ) -> None:
        assert repetitions is None or isinstance(repetitions, int)  # avoid silent issues
        self._repetitions = repetitions
        lchoices = list(choices)  # unroll iterables (includig Tuple instances
        if not lchoices:
            raise ValueError("{self._class__.__name__} received an empty list of options.")
        super().__init__(choices=container.Tuple(*lchoices), **kwargs)

    def __len__(self) -> int:
        """Number of choices"""
        return len(self.choices)

    def _get_parameters_str(self) -> str:
        params = sorted(
            (k, p.name)
            for k, p in self._content.items()
            if p.name != self._ignore_in_repr.get(k, "#ignoredrepr#")
        )
        return ",".join(f"{k}={n}" for k, n in params)

    @property
    def index(self) -> int:  # delayed choice
        """Index of the chosen option"""
        inds = self.indices.value
        assert inds.size == 1
        return int(inds[0])

    @property
    def indices(self) -> Array:
        """Array of indices of the chosen option"""
        return self["indices"]  # type: ignore

    @property
    def choices(self) -> container.Tuple:
        """The different options, as a Tuple Parameter"""
        return self["choices"]  # type: ignore

    def _layered_get_value(self) -> tp.Any:
        if self._repetitions is None:
            return core.as_parameter(self.choices[self.index]).value
        return tuple(core.as_parameter(self.choices[ind]).value for ind in self.indices.value)

    def _layered_set_value(self, value: tp.List[tp.Any]) -> None:
        """Must be adapted to each class
        This handles a list of values, not just one
        """  # TODO this is currenlty very messy, may need some improvement
        values = [value] if self._repetitions is None else value
        self._check_frozen()
        indices: np.ndarray = -1 * np.ones(len(values), dtype=int)
        # try to find where to put this
        for i, val in enumerate(values):
            for k, choice in enumerate(self.choices):
                try:
                    choice.value = val
                    indices[i] = k
                    break
                except Exception:  # pylint: disable=broad-except
                    pass
            if indices[i] == -1:
                raise ValueError(f"Could not figure out where to put value {value}")
        self.indices.value = indices

    def get_value_hash(self) -> tp.Hashable:
        hashes: tp.List[tp.Hashable] = []
        for ind in self.indices.value:
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
        lchoices = list(choices)
        rep = 1 if repetitions is None else repetitions
        indices = Array(shape=(rep, len(lchoices)), mutable_sigma=False)
        indices.add_layer(_datalayers.SoftmaxSampling(len(lchoices), deterministic=deterministic))
        super().__init__(
            choices=lchoices,
            repetitions=repetitions,
            indices=indices,
        )
        self._indices: tp.Optional[np.ndarray] = None

    def mutate(self) -> None:
        # force random_state sync
        self.random_state  # pylint: disable=pointless-statement
        self.indices.mutate()
        for ind in self.indices.value:
            self.choices[ind].mutate()


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
        indices = Array(init=len(choices) / 2.0 * np.ones((repetitions if repetitions is not None else 1,)))
        indices.set_bounds(0, len(choices), method="gaussian")
        indices = indices - 0.5
        intcasting = _datalayers.Int()
        intcasting.arity = len(choices)
        indices.add_layer(intcasting)
        super().__init__(
            choices=choices,
            repetitions=repetitions,
            indices=indices,
            transitions=transitions if isinstance(transitions, Array) else np.array(transitions, copy=False),
        )
        assert self.transitions.value.ndim == 1

    @property
    def transitions(self) -> Array:
        return self["transitions"]  # type: ignore

    def mutate(self) -> None:
        # force random_state sync
        self.random_state  # pylint: disable=pointless-statement
        transitions = core.as_parameter(self.transitions)
        transitions.mutate()
        rep = 1 if self._repetitions is None else self._repetitions
        #
        enc = discretization.Encoder(np.ones((rep, 1)) * np.log(self["transitions"].value), self.random_state)
        moves = enc.encode()
        signs = self.random_state.choice([-1, 1], size=rep)
        new_index = np.clip(self.indices.value + signs * moves, 0, len(self) - 1)
        self.indices.value = new_index
        # mutate corresponding parameter
        indices = set(self.indices.value)
        for ind in indices:
            self.choices[ind].mutate()
