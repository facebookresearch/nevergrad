# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
# importing NgDict to populate parameters (fake renaming for mypy explicit reimport)
# pylint: disable=unused-import,useless-import-alias
from . import discretization
from .core3 import Parameter
from .core3 import _as_parameter
from .core3 import NgDict as NgDict  # noqa


class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.

    Parameters
    ----------
    sigma: float or Array
        standard deviation of a mutation
    distribution: str
        distribution of the data ("linear" or "log")
    """

    def __init__(
            self,
            shape: t.Tuple[int, ...],
            sigma: t.Union[float, "Array"] = 1.0,
            distribution: t.Union[str, Parameter] = "linear",
            recombination: t.Union[str, Parameter] = "average"
    ) -> None:
        assert not isinstance(shape, Parameter)
        super().__init__(shape=shape, sigma=sigma, distribution=distribution, recombination=recombination)
        self._value: np.ndarray = np.zeros(shape)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray")
        if self._value.shape != value.shape:
            raise ValueError(f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}")
        self._value = value

    # pylint: disable=unused-argument
    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        sigma = self._get_parameter_value("sigma")
        self._value = (sigma * data).reshape(self.value.shape)

    def spawn_child(self) -> "Array":
        child = super().spawn_child()
        child._value = self.value
        return child

    def get_std_data(self) -> np.ndarray:
        sigma = self._get_parameter_value("sigma")
        reduced = self._value / sigma
        return reduced.ravel()  # type: ignore

    def recombine(self, *others: "Array") -> None:
        recomb = self._get_parameter_value("recombination")
        all_p = [self] + list(others)
        if recomb == "average":
            self.set_std_data(np.mean([p.get_std_data() for p in all_p], axis=0))
        else:
            raise ValueError(f'Unknown recombination "{recomb}"')


class NgTuple(NgDict):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, *parameters: t.Any) -> None:
        super().__init__()
        self._parameters.update({k: p for k, p in enumerate(parameters)})

    @property  # type: ignore
    def value(self) -> t.Tuple[t.Any, ...]:  # type: ignore
        param_val = [x[1] for x in sorted(self._parameters.items(), key=lambda x: int(x[0]))]
        return tuple(p.value if isinstance(p, Parameter) else p for p in param_val)

    @value.setter
    def value(self, value: t.Tuple[t.Any]) -> None:
        assert isinstance(value, tuple), "Value must be a tuple"
        for k, val in enumerate(value):
            _as_parameter(self[k]).value = val


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
        self._index_: t.Optional[int] = None

    @property
    def _index(self) -> int:  # delayed choice
        if self._index_ is None:
            self._draw(deterministic=self._deterministic)
        assert self._index_ is not None
        return self._index_

    @property
    def probabilities(self) -> Array:
        return self["probabilities"]  # type: ignore

    @property
    def choices(self) -> NgTuple:
        return self["choices"]  # type: ignore

    @property
    def value(self) -> t.Any:
        return _as_parameter(self.choices[self._index]).value

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
        self._index_ = index

    def get_value_hash(self) -> t.Hashable:
        return (self._index, _as_parameter(self.choices[self._index]).get_value_hash())

    def _draw(self, deterministic: bool = True) -> None:
        probas = self.probabilities.value
        random = False if deterministic or self._deterministic else self.random_state
        self._index_ = int(discretization.softmax_discretization(probas, probas.size, random=random)[0])

    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        super().set_std_data(data, deterministic=deterministic)
        self._draw(deterministic=deterministic)

    def mutate(self) -> None:
        self.probabilities.mutate()
        self._draw(deterministic=self._deterministic)
        param = self.choices[self._index]
        if isinstance(param, Parameter):
            param.mutate()

    def _internal_spawn_child(self: C) -> C:
        child = self.__class__(choices=[], deterministic=self._deterministic)
        child._parameters["choices"] = self.choices.spawn_child()
        child._parameters["probabilities"] = self.probabilities.spawn_child()
        child.parents_uids.append(self.uid)
        return child


class Instrumentation(NgTuple):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(NgTuple(*args), NgDict(**kwargs))

    @property
    def args(self) -> t.Tuple[t.Any, ...]:
        return self[0].value  # type: ignore

    @property
    def kwargs(self) -> t.Dict[str, t.Any]:
        return self[1].value  # type: ignore
