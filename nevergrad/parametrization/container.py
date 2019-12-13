# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
from .core import Parameter
from .core import _as_parameter
from .core import Dict as Dict  # Dict needs to be implemented in core since it's used in the base class


class Tuple(Dict):
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


class Instrumentation(Tuple):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super().__init__(Tuple(*args), Dict(**kwargs))

    @property
    def args(self) -> t.Tuple[t.Any, ...]:
        return self[0].value  # type: ignore

    @property
    def kwargs(self) -> t.Dict[str, t.Any]:
        return self[1].value  # type: ignore
