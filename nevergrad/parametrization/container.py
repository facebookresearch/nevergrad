# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from .core import Dict as Dict  # Dict needs to be implemented in core since it's used in the base class
from . import core


Ins = tp.TypeVar("Ins", bound="Instrumentation")
ArgsKwargs = tp.Tuple[tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any]]


class Tuple(Dict):
    """Tuple-valued parameter. This Parameter can contain other Parameters,
    its value is tuple which values are either directly the provided values
    if they are not Parameter instances, or the value of those Parameters.
    It also implements a getter to access the Parameters directly if need be.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the tuple

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the subparameters for all Parameter classes.
    """

    def __init__(self, *parameters: tp.Any) -> None:
        super().__init__()
        self._content.update({k: core.as_parameter(p) for k, p in enumerate(parameters)})
        self._sanity_check(list(self._content.values()))

    def _get_parameters_str(self) -> str:
        params = sorted((k, core.as_parameter(p).name) for k, p in self._content.items())
        return ",".join(f"{n}" for _, n in params)

    @property  # type: ignore
    def value(self) -> tp.Tuple[tp.Any, ...]:  # type: ignore
        param_val = [x[1] for x in sorted(self._content.items(), key=lambda x: int(x[0]))]
        return tuple(p.value if isinstance(p, core.Parameter) else p for p in param_val)

    @value.setter
    def value(self, value: tp.Tuple[tp.Any]) -> None:
        if not isinstance(value, tuple) or not len(value) == len(self):
            cls = self.__class__.__name__
            raise ValueError(f"{cls} value must be a tuple of size {len(self)}, got: {value}.\nCurrent value: {self.value}")
        for k, val in enumerate(value):
            core.as_parameter(self[k]).value = val


class Instrumentation(Tuple):
    """Container of parameters available through `args` and `kwargs` attributes.
    The parameter provided as input are used to provide values for
    an `arg` tuple and a `kwargs` dict.
    `value` attribue returns `(args, kwargs)`, but each can be independantly
    accessed through the `args` and `kwargs` properties.

    Parameters
    ----------
    *args
         values or Parameters to be used to fill the tuple of args
    *kwargs
         values or Parameters to be used to fill the dict of kwargs

    Note
    ----
    When used in conjonction with the "minimize" method of an optimizer,
    functions call use `func(*param.args, **param.kwargs)` instead of
    `func(param.value)`. This is for simplifying the parametrization of
    multiparameter functions.
    """

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__(Tuple(*args), Dict(**kwargs))
        self._sanity_check(list(self[0]._content.values()) + list(self[1]._content.values()))  # type: ignore

    @property
    def args(self) -> tp.Tuple[tp.Any, ...]:
        return self[0].value  # type: ignore

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        return self[1].value  # type: ignore
