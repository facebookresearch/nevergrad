# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Optional, Dict
import functools


class Registry(dict):
    """Registers function or classes as a dict.
    """

    def __init__(self) -> None:
        super().__init__()
        self._information: Dict[str, dict] = {}

    def register(self, obj: Any, info: Optional[Dict[Any, Any]] = None) -> Any:
        """Decorator method for registering functions/classes
        The info variable can be filled up using the register_with_info
        decorator instead of this one.
        """
        name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__
        if name in self:
            raise RuntimeError(f'Encountered a name collision "{name}"')
        self[name] = obj
        if info is not None:
            assert isinstance(info, dict)
            self._information[name] = info
        return obj

    def unregister(self, name: str) -> None:
        """Remove a previously-registered function or class, e.g. so you can
        re-register it in a Jupyter notebook.
        """
        if name in self:
            del self[name]

    def register_with_info(self, **info: Any) -> Callable:
        """Decorator for registering a function and information about it
        """
        return functools.partial(self.register, info=info)

    def get_info(self, name: str) -> dict:
        if name not in self:
            raise ValueError(f'"{name}" is not registered.')
        return self._information.setdefault(name, {})
