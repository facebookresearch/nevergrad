# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
from nevergrad.common.typetools import ArrayLike
from .core import Dict as Dict  # Dict needs to be implemented in core since it's used in the base class
from . import core


ArgsKwargs = t.Tuple[t.Tuple[t.Any, ...], t.Dict[str, t.Any]]


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

    def __init__(self, *parameters: t.Any) -> None:
        super().__init__()
        self._parameters.update({k: core.as_parameter(p) for k, p in enumerate(parameters)})

    def _get_parameters_str(self) -> str:
        params = sorted((k, core.as_parameter(p).name) for k, p in self._parameters.items())
        return ",".join(f"{n}" for _, n in params)

    @property  # type: ignore
    def value(self) -> t.Tuple[t.Any, ...]:  # type: ignore
        param_val = [x[1] for x in sorted(self._parameters.items(), key=lambda x: int(x[0]))]
        return tuple(p.value if isinstance(p, core.Parameter) else p for p in param_val)

    @value.setter
    def value(self, value: t.Tuple[t.Any]) -> None:
        assert isinstance(value, tuple), "Value must be a tuple"
        for k, val in enumerate(value):
            core.as_parameter(self[k]).value = val


class Instrumentation(Tuple):
    """Parameter holding args and kwargs:
    The parameter provided as input are used to provide values for
    an arg tuple and a kwargs dict.
    "value" attribue returns (args, kwargs), but each can be independantly
    accessed through the "args" and "kwargs" methods

    Parameters
    ----------
    *args
         values or Parameters to be used to fill the tuple of args
    *kwargs
         values or Parameters to be used to fill the dict of kwargs

    Note
    ----
    When used in conjonction with the "minimize" method of an optimizer,
    functions call use func(*param.args, **param.kwargs) instead of
    func(param.value). This is for simplifying the parametrization of
    multiparameter functions.
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(Tuple(*args), Dict(**kwargs))

    @property
    def args(self) -> t.Tuple[t.Any, ...]:
        return self[0].value  # type: ignore

    @property
    def kwargs(self) -> t.Dict[str, t.Any]:
        return self[1].value  # type: ignore

    # # # THE FOLLOWING IS ONLY FOR TEMPORARY (PARTIAL) COMPATIBILITY

    def cheap_constraint_check(self, *args: t.Any, **kwargs: t.Any) -> bool:
        child = self.spawn_child()
        child.value = (args, kwargs)
        return child.satisfies_constraint()

    @property
    def continuous(self) -> bool:
        return self.descriptors.continuous

    @property
    def noisy(self) -> bool:
        return not self.descriptors.deterministic

    def arguments_to_data(self, *args: t.Any, **kwargs: t.Any) -> np.ndarray:
        """Converts args and kwargs into data in np.ndarray format
        """
        child = self.spawn_child()
        child.value = (args, kwargs)  # type: ignore
        return child.get_std_data()

    def data_to_arguments(self, data: ArrayLike, deterministic: bool = False) -> ArgsKwargs:
        """Converts data to arguments
        Parameters
        ----------
        data: ArrayLike (list/tuple of floats, np.ndarray)
            the data in the optimization space
        deterministic: bool
            whether the conversion should be deterministic (some variables can be stochastic, if deterministic=True
            the most likely output will be used)
        Returns
        -------
        args: Tuple[Any]
            the positional arguments corresponding to the instance initialization positional arguments
        kwargs: Dict[str, Any]
            the keyword arguments corresponding to the instance initialization keyword arguments
        """
        child = self.spawn_child()
        child.set_std_data(np.array(data, copy=False), deterministic=deterministic)
        return child.value

# # # END OF COMPATIBILITY REQUIREMENT
