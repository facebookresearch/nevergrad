# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from nevergrad.parametrization import core
from nevergrad import optimizers
from nevergrad import callbacks


class Constraint(core.Operator):
    """Operator for applying a constraint on a Parameter

    Parameters
    ----------
    func: function
        the constraint function, taking the same arguments that the function to optimize.
        This constraint function must return a float (or a list/tuple/array of floats),
        positive if the constraint is not satisfied, null or negative otherwise.
    optimizer: str
        name of the optimizer to use for solving the constraint
    budget: int
        the budget to use for applying the constraint

    Example
    -------
    >>> constrained_parameter = Constraint(constraint_function)(parameter)
    >>> constrained_parameter.value  # value after trying to satisfy the constraint
    """

    _LAYER_LEVEL = core.Level.CONSTRAINT

    def __init__(self, func: tp.Callable[..., tp.Loss], optimizer: str = "NGOpt", budget: int = 100) -> None:
        super().__init__()
        self._func = func
        self._opt_cls = optimizers.registry[optimizer]
        self._budget = budget
        self._cache: tp.Any = None

    def _layered_del_value(self) -> None:
        self._cache = None  # clear cache!

    def apply_constraint(self, parameter: core.Parameter) -> core.Parameter:
        """Find a new parameter that better satisfies the constraint"""
        # This function can be overriden
        optim = self._opt_cls(parameter, budget=self._budget)
        early_stopping = callbacks.EarlyStopping(self.stopping_criterion)
        optim.register_callback("ask", early_stopping)
        optim.minimize(self.function)
        return optim.pareto_front()[0]

    def function(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss:
        out = self._func(*args, **kwargs)
        if isinstance(out, (bool, np.bool)):
            raise errors.NevergradTypeError(
                "Constraint must be a positive float if unsatisfied constraint (not bool)"
            )
        return np.maximum(0, out)  # type: ignore

    def parameter(self) -> core.Parameter:
        """Returns a constraint-free parameter, for the optimization process"""
        param = self._layers[0].copy()
        # remove last layer and make sure it is the last one
        if self._layer_index != param._layers.pop()._layer_index:
            raise RuntimeError("Constraint layer should be unique and placed last")
        return param  # type: ignore

    def stopping_criterion(self, optimizer: tp.Any) -> bool:
        """Checks whether a solution was found
        This is used as stopping criterio callback
        """
        if optimizer.num_tell < 1:
            return False
        best = optimizer.pareto_front()[0]
        return not np.any(best.losses > 0)

    def _layered_get_value(self) -> tp.Any:
        # TODO: this can be made more efficient (fewer copy) if need be.
        # Override only apply_constraint if you can, tampering with this method is tricky
        if self._cache is not None:
            return self._cache
        parameter = self.parameter()
        satisfied = not np.any(self.function(*parameter.args, **parameter.kwargs))
        if satisfied:
            self._cache = parameter.value
            return self._cache
        root: core.Parameter = self._layers[0]  # type: ignore
        recom = self.apply_constraint(parameter)
        root.set_standardized_data(np.zeros(root.dimension), reference=recom)
        self._cache = recom.value
        return self._cache
