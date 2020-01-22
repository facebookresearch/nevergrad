# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import typing as tp
import numpy as np
from nevergrad.common.typetools import ArrayLike
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
        assert isinstance(value, tuple), "Value must be a tuple"
        for k, val in enumerate(value):
            core.as_parameter(self[k]).value = val


class Instrumentation(Tuple):
    """Parameter holding args and kwargs:
    The parameter provided as input are used to provide values for
    an arg tuple and a kwargs dictp.
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

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__(Tuple(*args), Dict(**kwargs))
        self._sanity_check(list(self[0]._content.values()) + list(self[1]._content.values()))  # type: ignore
        self._compatibility_: tp.Optional["Instrumentation"] = None  # TODO remove when compatibility is over

    @property
    def args(self) -> tp.Tuple[tp.Any, ...]:
        return self[0].value  # type: ignore

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        return self[1].value  # type: ignore

    # # # THE FOLLOWING IS ONLY FOR TEMPORARY (PARTIAL) COMPATIBILITY

    def with_name(self: Ins, name: str) -> Ins:
        warnings.warn('"with_name" is deprecated, please use "set_name" instead', DeprecationWarning)
        return self.set_name(name)

    def cheap_constraint_check(self, *args: tp.Any, **kwargs: tp.Any) -> bool:
        warnings.warn("cheap_constraint_check(*args, **kwargs) is deprecated, use satisfies_constraints() instead",
                      DeprecationWarning)
        child = self.spawn_child()
        child.value = (args, kwargs)
        return child.satisfies_constraints()

    @property
    def continuous(self) -> bool:
        warnings.warn('"continuous" is deprecated, please use "descriptors.continuous" instead', DeprecationWarning)
        return self.descriptors.continuous

    @property
    def noisy(self) -> bool:
        warnings.warn('"noisy" is deprecated, please use "not descriptors.deterministic" instead', DeprecationWarning)
        return not self.descriptors.deterministic

    @property
    def probably_noisy(self) -> bool:
        warnings.warn('"probably_noisy" is deprecated, please use "not descriptors.deterministic_function" instead', DeprecationWarning)
        return not self.descriptors.deterministic_function

    @property
    def is_nonmetrizable(self) -> bool:
        warnings.warn('"is_nonmetrizable" is deprecated, please use "not descriptors.metrizable" instead', DeprecationWarning)
        return not self.descriptors.metrizable

    @property
    def _compatibility(self) -> "Instrumentation":
        if self._compatibility_ is None:
            self._compatibility_ = self.spawn_child()
        return self._compatibility_

    def arguments_to_data(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        """Converts args and kwargs into data in np.ndarray format
        """
        warnings.warn('"arguments_to_data" is deprecated, please use "get_standardized_data" instead', DeprecationWarning)
        self._compatibility.value = (args, kwargs)
        return self._compatibility.get_standardized_data(reference=self)

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
        warnings.warn('"data_to_arguments" is deprecated, please use "set_standardized_data" instead', DeprecationWarning)
        self._compatibility.set_standardized_data(np.array(data, copy=False), reference=self, deterministic=deterministic)
        return self._compatibility.value  # type: ignore

    def set_cheap_constraint_checker(self, func: tp.Callable[..., bool]) -> None:
        warnings.warn("set_cheap_constraint_checker is deprecated in favor of register_cheap_constraint with "
                      "slightly different API:\nthe registered function must only take one argument (the value"
                      " of the parameter, for an Instrumentation, this is a tuple countaining the tuple of args "
                      "and the dict of kwargs")
        self.register_cheap_constraint(FunctionPack(func))

    def get_summary(self, data: ArrayLike) -> str:
        raise RuntimeError("Summary is now suppressed because new parametrization is easier to analyze")


class FunctionPack:

    def __init__(self, func: tp.Callable[..., bool]) -> None:
        self.func = func

    def __call__(self, value: ArgsKwargs) -> bool:
        return self.func(*value[0], **value[1])


# # # END OF COMPATIBILITY REQUIREMENT
