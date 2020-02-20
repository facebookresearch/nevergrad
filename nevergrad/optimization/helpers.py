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
    cls: Optimizer-like object
        name of a nevergrad optimizer, or nevergrad optimizer class, or ConfiguredOptimizer instance
    module: torch.Module
        module which parameters need to be optimized
    bound: float
        values are clipped to [-bound, bound]

    Notes
    -----
    - This is experimental, the API may evolve
    - This does not support parallelization (multiple asks).
    """  # tested in functions.rl since torch is only loaded there

    def __init__(
        self,
        cls: tp.Union[str, Optim],
        module: tp.Any,  # torch is not typed
        bound: float = 20.0,
    ) -> None:
        self.module = module
        kwargs = {
            name: ng.p.Array(init=value).set_bounds(-bound, bound, method="clipping")
            for name, value in module.state_dict().items()
        }  # bounded to avoid overflows
        if isinstance(cls, str):
            cls = optimizerlib.registry[cls]
        self.optimizer = cls(ng.p.Dict(**kwargs), budget=None, num_workers=1)
        self.candidate = self.optimizer.ask()

    def _set_candidate(self) -> None:
        # pylint:disable=not-callable
        state = {x: torch.tensor(y.astype(np.float32)) for x, y in self.candidate.value.items()}
        self.module.load_state_dict(state)

    def step(self, loss: float) -> None:
        self.optimizer.tell(self.candidate, loss)
        self.candidate = self.optimizer.ask()
        self._set_candidate()
