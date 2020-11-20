# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import functools


X = tp.TypeVar("X")


# pylint does not understand Dict[str, X],
# so we reimplement the MutableMapping interface
class Registry(tp.MutableMapping[str, X]):
    """Registers function or classes as a dict.
    """

    def __init__(self) -> None:
        super().__init__()
        self.data: tp.Dict[str, X] = {}
        self._information: tp.Dict[str, tp.Dict[tp.Hashable, tp.Any]] = {}

    def register(self, obj: X, info: tp.Optional[tp.Dict[tp.Hashable, tp.Any]] = None) -> X:
        """Decorator method for registering functions/classes
        The info variable can be filled up using the register_with_info
        decorator instead of this one.
        """
        name = getattr(obj, "__name__", obj.__class__.__name__)
        self.register_name(name, obj, info)
        return obj

    def register_name(self, name: str, obj: X, info: tp.Optional[tp.Dict[tp.Hashable, tp.Any]] = None) -> None:
        """Register an object with a provided name
        """
        if name in self:
            raise RuntimeError(f'Encountered a name collision "{name}"')
        self[name] = obj
        if info is not None:
            assert isinstance(info, dict)
            self._information[name] = info

    def unregister(self, name: str) -> None:
        """Remove a previously-registered function or class, e.g. so you can
        re-register it in a Jupyter notebook.
        """
        if name in self:
            del self[name]

    def register_with_info(self, **info: tp.Any) -> tp.Callable[[X], X]:
        """Decorator for registering a function and information about it
        """
        return functools.partial(self.register, info=info)

    def get_info(self, name: str) -> tp.Dict[tp.Hashable, tp.Any]:
        if name not in self:
            raise ValueError(f'"{name}" is not registered.')
        return self._information.setdefault(name, {})

    def __getitem__(self, key: str) -> X:
        return self.data[key]

    def __setitem__(self, key: str, value: X) -> None:
        self.data[key] = value

    def __delitem__(self, key: str) -> None:
        del self.data[key]

    def __iter__(self) -> tp.Iterator[str]:
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)
