# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad as ng
from .. import base

# pylint: disable=import-outside-toplevel


class PBOFunction(base.ExperimentFunction):
    """
    Pseudo-boolean functions taken from the IOHexperimenter project, adapted for minimization

    Parameters
    ----------
    fid: int
        function number (for a list of functions, please see the
        `documentation <https://www.sciencedirect.com/science/article/abs/pii/S1568494619308099>>`_
    iid: int
        the instance of the function, specifies the transformations in variable and objective space
    dim: int
        The dimensionality of the problem. For function 21 and 23 (N-queens), this needs to be a perfect square
    instrumentation: str
        How the paramerterization is handled. Either "Softmax", in which case the optimizers will work on probabilities
        (so the same parametrization may actually lead to different values), or "Ordered". With this one the optimizer
        will handle a value, which will lead to a 0 if it is negative, and 1 if it is positive.
        This way it is no more stochastic, but the problem becomes very non-continuous

    Notes
    -----
    The IOHexperimenter is part of the IOHprofiler project (IOHprofiler.github.io)

    References
    -----
    For a full description of the available problems and used transformation methods,
    please see "Benchmarking discrete optimization heuristics with IOHprofiler" by C. Doerr et al.
    """

    def __init__(self, fid: int = 1, iid: int = 0, dim: int = 16, instrumentation: str = "Softmax") -> None:
        from IOHexperimenter import IOH_function  # lazy import in case it is not installed

        if fid in [21, 23]:
            assert np.sqrt(
                dim
            ).is_integer(), "Dimension needs to be a perfect square for the selected problem"
        self.f_internal = IOH_function(fid=fid, dim=dim, iid=iid, suite="PBO")
        assert instrumentation in [
            "Softmax",
            "Ordered",
        ], "The only valid options for 'instrumentation' are 'Softmax' and 'Ordered'"
        if instrumentation == "Softmax":
            parameterization: np.p.Parameter = ng.p.Choice([0, 1], repetitions=dim)
        else:
            parameterization = ng.p.TransitionChoice([0, 1], repetitions=dim)
        super().__init__(self._evaluation_internal, parameterization.set_name(instrumentation))

    def _evaluation_internal(self, x: np.ndarray) -> float:
        assert len(x) == self.f_internal.number_of_variables
        return -float(self.f_internal(x))


class WModelFunction(base.ExperimentFunction):
    """
    Implementation of the W-Model taken from the IOHexperimenter project, adapted for minimization
    Currently supports only OneMax and LeadingOnes as base-functions

    Parameters
    ----------
    base_function: str
        Which base function to use. Either "OneMax" or "LeadingOnes"
    iid: int
        the instance of the function, specifies the transformations in variable and objective space
    dim: int
        The dimensionality of the problem.
    dummy:
        Float between 0 and 1, fractoin of valid bits.
    epistasis:
        size of sub-string for epistasis
    neutrality:
        size of sub-string for neutrality
    ruggedness:
        gamma for ruggedness layper
    instrumentation: str
        How the paramerterization is handled. Either "Softmax", in which case the optimizers will work on probabilities
        (so the same parametrization may actually lead to different values), or "Ordered". With this one the optimizer
        will handle a value, which will lead to a 0 if it is negative, and 1 if it is positive.
        This way it is no more stochastic, but the problem becomes very non-continuous

    Notes
    -----
    The IOHexperimenter is part of the IOHprofiler project (IOHprofiler.github.io)

    References
    -----
    For a description of the W-model in the IOHexperimenter and of the used transformation methods,
    please see "Benchmarking discrete optimization heuristics with IOHprofiler" by C. Doerr et al.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        base_function: str = "OneMax",
        iid: int = 0,
        dim: int = 16,
        dummy: float = 0,
        epistasis: int = 0,
        neutrality: int = 0,
        ruggedness: int = 0,
        instrumentation: str = "Softmax",
    ) -> None:
        from IOHexperimenter import W_model_function  # lazy import in case it is not installed

        assert epistasis <= dim, "Epistasis has to be less or equal to than dimension"
        assert neutrality <= dim, "Neutrality has to be less than or equal to dimension"
        assert ruggedness <= dim ** 2, "Ruggedness has to be less than or equal to dimension squared"
        assert 0 <= dummy <= 1, "Dummy variable fraction has to be in [0,1]"
        self.f_internal = W_model_function(
            base_function=base_function,
            iid=iid,
            dim=dim,
            dummy=dummy,
            epistasis=epistasis,
            neutrality=neutrality,
            ruggedness=ruggedness,
        )
        assert instrumentation in [
            "Softmax",
            "Ordered",
        ], "The only valid options for 'instrumentation' are 'Softmax' and 'Unordered'"
        if instrumentation == "Softmax":
            parameterization: ng.p.Parameter = ng.p.Choice([0, 1], repetitions=dim)
        else:
            parameterization = ng.p.TransitionChoice([0, 1], repetitions=dim)
        super().__init__(self._evaluation_internal, parameterization.set_name("instrumentation"))

    def _evaluation_internal(self, x: np.ndarray) -> float:
        assert len(x) == self.f_internal.number_of_variables
        assert not set(x) - {0, 1}, f"Pb with input {x} in PBO: not binary."
        return -float(self.f_internal(x))
