# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import inspect
import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.common.tools import pytorch_import_fix
from . import base
from . import optimizerlib
pytorch_import_fix()

# pylint: disable=wrong-import-position,wrong-import-order
try:
    import torch
except ImportError:
    pass  # experimental code, torch is not necessarily imported in this part of the package

Optim = tp.Union[tp.Type[base.Optimizer], base.ConfiguredOptimizer]


class TorchOptimizer:
    """Experimental helper to perform optimization using torch
    workflow with a nevergrad optimizer

    Parameters
    ----------
    parameters: iterable
        module parameters which need to be optimized
    cls: Optimizer-like object
        name of a nevergrad optimizer, or nevergrad optimizer class, or ConfiguredOptimizer instance
    bound: float
        values are clipped to [-bound, bound]

    Notes
    -----
    - This is experimental, the API may evolve
    - This does not support parallelization (multiple asks).


    Example
    -------
    ..code::python

        module = ...
        optimizer = helpers.TorchOptimizer(module.parameters(), "OnePlusOne")
        for x, y in batcher():
            loss = compute_loss(module(x), y)
            optimizer.step(loss)

    """  # tested in functions.rl since torch is only loaded there

    def __init__(
        self,
        parameters: tp.Iterable[tp.Any],  # torch is not typed
        cls: tp.Union[str, Optim],
        bound: float = 20.0,
    ) -> None:
        self.parameters = list(parameters)
        if isinstance(cls, str):
            cls = optimizerlib.registry[cls]
        elif not isinstance(cls, base.ConfiguredOptimizer):
            if not (inspect.isclass(cls) and issubclass(cls, base.Optimizer)):
                raise TypeError('"cls" must be a str, a ConfiguredOptimizer instance, or an Optimizer class')
        args = (
            ng.p.Array(init=np.array(p.data, dtype=np.float)).set_bounds(-bound, bound, method="clipping")
            for p in self.parameters
        )  # bounded to avoid overflows
        self.optimizer = cls(ng.p.Tuple(*args), budget=None, num_workers=1)
        self.candidate = self.optimizer.ask()

    def _set_candidate(self) -> None:
        for p, data in zip(self.parameters, self.candidate.value):
            # pylint:disable=not-callable
            p.data = torch.tensor(data.astype(np.float32))

    def step(self, loss: float) -> None:
        self.optimizer.tell(self.candidate, loss)
        self.candidate = self.optimizer.ask()
        self._set_candidate()
