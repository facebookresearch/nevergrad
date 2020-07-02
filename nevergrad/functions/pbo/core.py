# from IOHexperimenter import IOH_function 
#TODO: figure out which way is better. importing the pbo_dict ignores all wrapper-functions, but it seems
#like they are not needed here anyway, only the evaluate is actually used, so this should be faster
#But, is is quite an ugly solution and should probably be improved
from IOHexperimenter.src.IOH_function import pbo_fid_dict 
from IOHexperimenter import W_Model_OneMax, W_Model_LeadingOnes
import numpy as np
import nevergrad as ng
from nevergrad.common.typetools import ArrayLike
from .. import base

class PBOFunction(base.ExperimentFunction):
    """
    Pseudo-boolean functions taken from the IOHexperimenter project, adapted for minimization
    
    Parameters
    ----------
    fid: int
        function number (for a list of functions, please see the documentation)
    iid: int
        the instance of the function, specifies the transformations in variable and objective space
    dim: int
        The dimensionality of the problem. For function 21 and 23 (N-queens), this needs to be a perfect square
    instrum_str: str
        How the paramerterization is handled. Either "Softmax", in which case the optimizers will work on probabilities
        (so the same parametrization may actually lead to different values), or "Unordered". With this one the optimizer 
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

    def __init__(self, fid: int = 1, iid: int = 0, dim: int = 16, instrum_str: str = "Softmax") -> None:
        if fid in [21, 23]:
            assert np.sqrt(dim).is_integer(), "Dimension needs to be a perfect square for the selected problem"
#         self.f_internal = IOH_function(fid = fid, dim = dim, iid = iid, suite = "PBO")
        self.f_internal = pbo_fid_dict[fid](iid, dim)
        assert instrum_str in ["Softmax", "Unordered"], "The only valid options for 'instrum_str' are 'Softmax' and 'Unordered'"
        if instrum_str == "Softmax":
            parameterization = ng.p.Choice([0, 1], repetitions=dim)
        else:
            parameterization = ng.p.TransitionChoice([0, 1], repetitions=dim)
        super().__init__(self._evaluation_internal, parameterization)
        self.descriptors.update(fid = fid, iid = iid)
        self.register_initialization()


    def _evaluation_internal(self, x: np.ndarray) -> float:
        assert len(x) == self.dimension, f"Expected dimension {self.dimension}, got {len(x)}"
        return -1 * self.f_internal.evaluate(x)


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
    instrum_str: str
        How the paramerterization is handled. Either "Softmax", in which case the optimizers will work on probabilities
        (so the same parametrization may actually lead to different values), or "Unordered". With this one the optimizer 
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

    def __init__(self, base_function: str = "OneMax", iid: int = 0, dim: int = 16, dummy: float = 0, epistasis: int = 0,
                 neutrality: int = 0, ruggedness: int = 0, instrum_str: str = "Softmax") -> None:
        assert epistasis <= dim, "Epistasis has to be less or equal to than dimension"
        assert neutrality <= dim, "Neutrality has to be less than or equal to dimension"
        assert ruggedness <= dim ** 2, "Ruggedness has to be less than or equal to dimension squared"
        assert dummy <= 1 and dummy >= 0, "Dummy variable fraction has to be in [0,1]"
        
        if base_function == "OneMax":
            self.f_internal = W_Model_OneMax(iid, dim)
        elif base_function == "LeadingOnes":
            self.f_internal = W_Model_LeadingOnes(iid, dim)
        self.f_internal.set_w_setting(dummy, epistasis, neutrality, ruggedness)
        assert instrum_str in ["Softmax", "Unordered"], "The only valid options for 'instrum_str' are 'Softmax' and 'Unordered'"
        if instrum_str == "Softmax":
            parameterization = ng.p.Choice([0, 1], repetitions=dim)
        else:
            parameterization = ng.p.TransitionChoice([0, 1], repetitions=dim)
        super().__init__(self._evaluation_internal, parameterization)
        self.descriptors.update(base_function = base_function, iid = iid, dummy = dummy, epistasis = epistasis, 
                                neutrality = neutrality, ruggedness = ruggedness)
        self.register_initialization()


    def _evaluation_internal(self, x: np.ndarray) -> float:
        assert len(x) == self.dimension, f"Expected dimension {self.dimension}, got {len(x)}"
        return -1 * self.f_internal.evaluate(x)