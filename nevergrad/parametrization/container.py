# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import numpy as np
import nevergrad.common.typing as tp
from . import utils
from . import core


D = tp.TypeVar("D", bound="Container")


class Container(core.Parameter):
    """Parameter which can hold other parameters.
    This abstract implementation is based on a dictionary.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the dict

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the internal/model parameters for all Parameter classes.
    """

    def __init__(self, **parameters: tp.Any) -> None:
        super().__init__()
        self._subobjects = utils.Subobjects(self, base=core.Parameter, attribute="_content")
        self._content: tp.Dict[tp.Any, core.Parameter] = {
            k: core.as_parameter(p) for k, p in parameters.items()
        }
        self._sizes: tp.Optional[tp.Dict[str, int]] = None
        self._sanity_check(list(self._content.values()))
        self._ignore_in_repr: tp.Dict[
            str, str
        ] = {}  # hacky undocumented way to bypass boring representations

    @property
    def dimension(self) -> int:
        return sum(x.dimension for x in self._content.values())

    def _sanity_check(self, parameters: tp.List[core.Parameter]) -> None:
        """Check that all parameters are different"""
        # TODO: this is first order, in practice we would need to test all the different
        # parameter levels together
        if parameters:
            assert all(isinstance(p, core.Parameter) for p in parameters)
            ids = {id(p) for p in parameters}
            if len(ids) != len(parameters):
                raise ValueError("Don't repeat twice the same parameter")

    def __getitem__(self, name: tp.Any) -> core.Parameter:
        return self._content[name]

    def __len__(self) -> int:
        return len(self._content)

    def _get_parameters_str(self) -> str:
        raise NotImplementedError

    def _get_name(self) -> str:
        return f"{self.__class__.__name__}({self._get_parameters_str()})"

    def get_value_hash(self) -> tp.Hashable:
        return tuple(sorted((x, y.get_value_hash()) for x, y in self._content.items()))

    def _internal_get_standardized_data(self: D, reference: D) -> np.ndarray:
        data = {k: self[k].get_standardized_data(reference=p) for k, p in reference._content.items()}
        if self._sizes is None:
            self._sizes = OrderedDict(sorted((x, y.size) for x, y in data.items()))
        assert self._sizes is not None
        data_list = [data[k] for k in self._sizes]
        if not data_list:
            return np.array([])
        return data_list[0] if len(data_list) == 1 else np.concatenate(data_list)  # type: ignore

    def _internal_set_standardized_data(self: D, data: np.ndarray, reference: D) -> None:
        if self._sizes is None:
            self.get_standardized_data(reference=self)
        assert self._sizes is not None
        if data.size != sum(v for v in self._sizes.values()):
            raise ValueError(
                f"Unexpected shape {data.shape} for {self} with dimension {self.dimension}:\n{data}"
            )
        data = data.ravel()
        start, end = 0, 0
        for name, size in self._sizes.items():
            end = start + size
            self._content[name].set_standardized_data(data[start:end], reference=reference[name])
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"

    def _layered_sample(self: D) -> D:
        child = self.spawn_child()
        child._content = {k: p.sample() for k, p in self._content.items()}
        return child


class Dict(Container):
    """Dictionary-valued parameter. This Parameter can contain other Parameters,
    its value is a dict, with keys the ones provided as input, and corresponding values are
    either directly the provided values if they are not Parameter instances, or the value of those
    Parameters. It also implements a getter to access the Parameters directly if need be.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the dict

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the internal/model parameters for all Parameter classes.
    """

    value: core.ValueProperty[tp.Dict[str, tp.Any], tp.Dict[str, tp.Any]] = core.ValueProperty()

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self.keys())

    def keys(self) -> tp.KeysView[str]:
        return self._content.keys()

    def items(self) -> tp.ItemsView[str, core.Parameter]:
        return self._content.items()

    def values(self) -> tp.ValuesView[core.Parameter]:
        return self._content.values()

    def _layered_get_value(self) -> tp.Dict[str, tp.Any]:
        return {k: p.value for k, p in self.items()}

    def _layered_set_value(self, value: tp.Dict[str, tp.Any]) -> None:
        cls = self.__class__.__name__
        if not isinstance(value, dict):
            raise TypeError(f"{cls} value must be a dict, got: {value}\nCurrent value: {self.value}")
        if set(value) != set(self):
            raise ValueError(
                f"Got input keys {set(value)} for {cls} but expected {set(self._content)}\nCurrent value: {self.value}"
            )
        for key, val in value.items():
            self._content[key].value = val

    def _get_parameters_str(self) -> str:
        params = sorted(
            (k, p.name) for k, p in self.items() if p.name != self._ignore_in_repr.get(k, "#ignoredrepr#")
        )
        return ",".join(f"{k}={n}" for k, n in params)


class Tuple(Container):
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
        return ",".join(f"{param.name}" for param in self)

    def __iter__(self) -> tp.Iterator[core.Parameter]:
        return (self._content[k] for k in range(len(self)))

    value: core.ValueProperty[tp.Tuple[tp.Any, ...], tp.Tuple[tp.Any, ...]] = core.ValueProperty()

    def _layered_get_value(self) -> tp.Tuple[tp.Any, ...]:
        return tuple(p.value for p in self)

    def _layered_set_value(self, value: tp.Tuple[tp.Any, ...]) -> None:
        if not isinstance(value, tuple) or not len(value) == len(self):
            cls = self.__class__.__name__
            raise ValueError(
                f"{cls} value must be a tuple of size {len(self)}, got: {value}.\nCurrent value: {self.value}"
            )
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
        return self.value[0]

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        return self.value[1]

    value: core.ValueProperty[tp.ArgsKwargs, tp.ArgsKwargs] = core.ValueProperty()  # type: ignore
