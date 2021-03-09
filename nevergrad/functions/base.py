# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from pathlib import Path
import numbers
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from nevergrad.common.errors import (  # pylint: disable=unused-import
    UnsupportedExperiment as UnsupportedExperiment,
)
from nevergrad.parametrization import parameter as p
from nevergrad.optimization import multiobjective as mobj

EF = tp.TypeVar("EF", bound="ExperimentFunction")
ME = tp.TypeVar("ME", bound="MultiExperiment")


def _reset_copy(obj: p.Parameter) -> p.Parameter:
    """Copy a parameter and resets its random state to obtain variability"""
    out = obj.copy()
    out.random_state = None
    return out


# pylint: disable=too-many-instance-attributes
class ExperimentFunction:
    """Combines a function and its parametrization for running experiments (see benchmark subpackage)

    Parameters
    ----------
    function: callable
        the callable to convert
    parametrization: Parameter
        the parametrization of the function
    Notes
    -----
    - you can redefine custom "evaluation_function" and "compute_pseudotime" for custom behaviors in experiments
    - the bool/int/str/float init arguments are added as descriptors for the experiment which will serve in
      definining test cases. You can add more through "add_descriptors".
    - Makes sure you the "copy()" methods works (provides a new copy of the function *and* its parametrization)
      if you subclass ExperimentFunction since it is intensively used in benchmarks.
      By default, this will create a new instance using the same init arguments as your current instance
      (they were recorded through "__new__"'s magic) and apply the additional descriptors you may have added,
      as well as propagate the new parametrization *if it has a different name as the current one*.
    """

    def __new__(cls: tp.Type[EF], *args: tp.Any, **kwargs: tp.Any) -> EF:
        """Identifies initialization parameters during initialization and store them"""
        inst = object.__new__(cls)
        sig = inspect.signature(cls.__init__)
        callargs: tp.Dict[str, tp.Any] = {}
        try:
            boundargs = sig.bind(inst, *args, **kwargs)
        except TypeError:
            pass  # either a problem which will be caught later or a unpickling
        else:
            boundargs.apply_defaults()  # make sure we get the default non-provided arguments
            callargs = dict(boundargs.arguments)
            callargs.pop("self")
        inst._auto_init = callargs
        inst._descriptors = {
            x: y for x, y in callargs.items() if isinstance(y, (str, tuple, int, float, bool))
        }
        inst._descriptors["function_class"] = cls.__name__
        return inst  # type: ignore

    def __init__(
        self: EF,
        function: tp.Callable[..., tp.Loss],
        parametrization: p.Parameter,
    ) -> None:
        assert callable(function)
        assert not hasattr(
            self, "_initialization_kwargs"
        ), '"register_initialization" was called before super().__init__'
        self._auto_init: tp.Dict[str, tp.Any]  # filled by __new__
        self._descriptors: tp.Dict[str, tp.Any]  # filled by __new__
        self._parametrization: p.Parameter
        self.parametrization = parametrization
        # force random state initialization
        self.multiobjective_upper_bounds: tp.Optional[np.ndarray] = None
        self.__function = function  # __ to prevent overrides
        # if this is not a function bound to this very instance, add the function/callable name to the descriptors
        if not hasattr(function, "__self__") or function.__self__ != self:  # type: ignore
            name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
            self._descriptors.update(name=name)
        if len(self.parametrization.name) > 24:
            raise RuntimeError(
                f"For the sake of benchmarking, please rename the current parametrization:\n{self.parametrization!r}\n"
                "to a shorter name. This way it will be more readable in the experiments.\n"
                'Eg: parametrization.set_name("") to just ignore it\n'
                "CAUTION: Make sure you set different names for different parametrization configurations if you want it "
                "to be used in order to differentiate between benchmarks cases."
            )

    @property
    def dimension(self) -> int:
        return self._parametrization.dimension

    @property
    def parametrization(self) -> p.Parameter:
        return self._parametrization

    @parametrization.setter
    def parametrization(self, parametrization: p.Parameter) -> None:
        self._parametrization = parametrization
        self._parametrization.freeze()
        # pylint: disable=pointless-statement
        self._parametrization.random_state  # force initialization for synchronization of random state
        # # TODO investigate why this synchronization is needed

    @property
    def function(self) -> tp.Callable[..., tp.Loss]:
        return self.__function

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Loss:
        """Call the function directly (equivaluent to parametrized_function.function(*args, **kwargs))"""
        return self.function(*args, **kwargs)

    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
        noise_level, transform and dimension
        """
        desc = dict(self._descriptors)  # Avoid external modification
        desc.update(parametrization=self.parametrization.name, dimension=self.dimension)
        return desc

    def add_descriptors(self, **kwargs: tp.Optional[tp.Hashable]) -> None:
        self._descriptors.update(kwargs)

    def __repr__(self) -> str:
        """Shows the function name and its summary"""
        params = [f"{x}={repr(y)}" for x, y in sorted(self._descriptors.items())]
        return "Instance of {}({})".format(self.__class__.__name__, ", ".join(params))

    def equivalent_to(self, other: tp.Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal
        (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
        if other.__class__ != self.__class__:
            return False
        return (
            bool(self._descriptors == other._descriptors)
            and self.parametrization.name == other.parametrization.name
        )

    def _internal_copy(self: EF) -> EF:
        """This is "black magic" which creates a new instance using the same init parameters
        that you provided and which were recorded through the __new__ method of ExperimentFunction
        """
        # auto_init is automatically filled by __new__, aka when creating the instance
        output: EF = self.__class__(
            **{x: _reset_copy(y) if isinstance(y, p.Parameter) else y for x, y in self._auto_init.items()}
        )
        return output

    def copy(self: EF) -> EF:
        """Provides a new equivalent instance of the class, possibly with
        different random initialization, to provide different equivalent test cases
        when using different seeds.
        This also checks that parametrization and descriptors are correct.
        You should preferably override _internal_copy
        """
        # add descriptors present in self but not added by initialization
        # (they must have been added manually)
        output = self._internal_copy()
        keys = set(output.descriptors)
        output.add_descriptors(**{x: y for x, y in self.descriptors.items() if x not in keys})
        # parametrization may have been overriden, so let's always update it
        # Caution: only if names differ!
        if output.parametrization.name != self.parametrization.name:
            output.parametrization = _reset_copy(self.parametrization)
        # then if there are still differences, something went wrong
        if not output.equivalent_to(self):
            raise errors.ExperimentFunctionCopyError(
                f"Copy of\n{self}\nwith descriptors:\n{self._descriptors}\nreturned non-equivalent\n"
                f"{output}\nwith descriptors\n{output._descriptors}.\n\n"
                "This means that the auto-copy behavior of ExperimentFunction does not work.\n"
                "You may want to implement your own copy method, or check implementation of "
                "ExperimentFunction.__new__ and copy to better understand what happens"
            )
        # propagate other useful information # TODO a bit hacky
        output.parametrization._constraint_checkers = self.parametrization._constraint_checkers
        output.multiobjective_upper_bounds = (
            self.multiobjective_upper_bounds
        )  # TODO not sure why this is needed
        return output

    def compute_pseudotime(  # pylint: disable=unused-argument
        self, input_parameter: tp.Any, loss: tp.Loss
    ) -> float:
        """Computes a pseudotime used during benchmarks for mocking parallelization in a reproducible way.
        By default, each call takes 1 unit of pseudotime, but this can be modified by overriding this
        function and the pseudo time can be a function of the function inputs and output.

        Note: This replaces get_postponing_delay which has been aggressively deprecated

        Parameters
        ----------
        input_parameter: Any
            the input that was provided to the actual function
        value: float
            the output of the actual function

        Returns
        -------
        float
            the pseudo computation time of the call to the actual function
        """
        return 1.0

    def evaluation_function(self, *recommendations: p.Parameter) -> float:
        """Provides the evaluation crieterion for the experiment.
        In case of mono-objective, it defers to evaluation_function
        Otherwise, it uses the hypervolume.
        This function can be overriden to provide custom behaviors.

        Parameters
        ----------
        *pareto: Parameter
            pareto front provided by the optimizer
        """

        if self.multiobjective_upper_bounds is None:  # monoobjective case
            assert len(recommendations) == 1
            output = self.function(*recommendations[0].args, **recommendations[0].kwargs)
            assert isinstance(
                output, numbers.Number
            ), f"evaluation_function can only be called on monoobjective experiments (output={output}) function={self.function}."
            return output
        # multiobjective case
        hypervolume = mobj.HypervolumePareto(
            upper_bounds=self.multiobjective_upper_bounds, seed=self.parametrization.random_state
        )
        for candidate in recommendations:
            hypervolume.add(candidate)
        return -hypervolume.best_volume


def update_leaderboard(identifier: str, loss: float, array: np.ndarray, verbose: bool = True) -> None:
    """Handy function for storing best results for challenging functions (eg.: Photonics)
    The best results are kept in a file that is automatically updated with new data.
    This may require installing nevergrad in dev mode.

    Parameters
    ----------
    identifier: str
        the identifier of the problem
    loss: float
        the new loss, if better than the one in the file, the file will be updated
    array: np.ndarray
        the array corresponding to the loss
    verbose: bool
        whether to also print a message if the leaderboard was updated
    """
    # pylint: disable=import-outside-toplevel
    import pandas as pd  # lazzy to avoid requiring pandas for using an ExperimentFunction

    loss = np.round(loss, decimals=12)  # this is probably already too precise for the machine
    filepath = Path(__file__).with_name("leaderboard.csv")
    bests = pd.DataFrame(columns=["loss", "array"])
    if filepath.exists():
        bests = pd.read_csv(filepath, index_col=0)
    if identifier not in bests.index:
        bests.loc[identifier, :] = (float("inf"), "")
    try:
        if not bests.loc[identifier, "loss"] < loss:  # works for nan
            bests.loc[identifier, "loss"] = loss
            string = "[" + ",".join(str(x) for x in array.ravel()) + "]"
            bests.loc[identifier, "array"] = string
            bests = bests.loc[sorted(x for x in bests.index), :]
            bests.to_csv(filepath)
            if verbose:
                print(f"New best value for {identifier}: {loss}\nwith: {string[:80]}")
    except Exception:  # pylint: disable=broad-except
        pass  # better avoir bugs for this


class MultiExperiment(ExperimentFunction):
    """Pack several mono-objective experiments into a multiobjective experiment


    Parameters
    ----------
    experiments: iterable of ExperimentFunction

    Notes
    -----
    - packing of multiobjective experiments is not supported.
    - parametrization must match between all functions (only their name is checked as initialization)
    - there is no descriptor for the packed functions, except the name (concatenetion of packed function names).
    """

    def __init__(
        self,
        experiments: tp.Iterable[ExperimentFunction],
        upper_bounds: tp.ArrayLike,
    ) -> None:
        xps = list(experiments)
        assert xps
        assert len(xps) == len({id(xp) for xp in xps}), "All experiments must be different instances"
        assert all(
            xp.multiobjective_upper_bounds is None for xp in xps
        ), "Packing multiobjective xps is not supported."
        assert all(
            xps[0].parametrization.name == xp.parametrization.name for xp in xps[1:]
        ), "Parametrization do not match"
        super().__init__(self._multi_func, xps[0].parametrization)
        self.multiobjective_upper_bounds = np.array(upper_bounds)
        self._descriptors.update(name=",".join(xp._descriptors.get("name", "#unknown#") for xp in xps))
        self._experiments = xps

    def _multi_func(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        outputs = [f(*args, **kwargs) for f in self._experiments]
        return np.array(outputs)

    def _internal_copy(self) -> "MultiExperiment":
        assert self.multiobjective_upper_bounds is not None
        return MultiExperiment([f.copy() for f in self._experiments], self.multiobjective_upper_bounds)
