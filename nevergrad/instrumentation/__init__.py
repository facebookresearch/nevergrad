# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from nevergrad.parametrization.utils import TemporaryDirectoryCopy, CommandFunction
from . import variables
from . import variables as var
from .multivariables import Instrumentation as Instrumentation
from .multivariables import InstrumentedFunction as InstrumentedFunction


class FolderFunction:

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        raise RuntimeError("Please import as nevergrad.parametrization.FolderFunction")


__all__ = ["Instrumentation", "var", "CommandFunction", "FolderFunction"]
