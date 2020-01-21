# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import
import typing as tp
from ..parametrization.parameter import Instrumentation as Instrumentation


class InstrumentedFunction:
    """InstrumentedFunction is being aggressively deprecated. Conversion depends on your use case:
    - for optimization purpose: directly provide ng.Instrumentation to the optimizer, it will
       provide candidates with fields 'args' and 'kwargs' that match the instrumentation.
    - for benchmarks: derive from ng.functions.ExperimentFunction. Main differences are:
       calls to __call__ directly forwards the main function (instead of converting from data space),
       __init__ takes exactly two arguments (main function and parametrization/instrumentation) and
      instrumentation is renamed to parametrization for forward compatibility.
    """

    def __init__(self, function: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        raise RuntimeError(self.__doc__)
