# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from pathlib import Path
import numpy as np
from nevergrad.parametrization import parameter as p


EF = tp.TypeVar("EF", bound="ExperimentFunction")


class ExperimentFunctionCopyError(NotImplementedError):
    """Raised when the experiment function fails to copy itself (for benchmarks)
    """


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
    - You can update the "_descriptors" dict attribute so that function parameterization is recorded during benchmark
    - Makes sure you the "copy()" methods works (provides a new copy of the function *and* its parametrization)
      if you subclass ExperimentFunction since it is intensively used in benchmarks.
    """

    def __init__(self: EF, function: tp.Callable[..., float], parametrization: p.Parameter) -> None:
        assert callable(function)
        assert not hasattr(self, "_initialization_kwargs"), '"register_initialization" was called before super().__init__'
        self._initialization_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None
        self._initialization_func: tp.Callable[..., EF] = self.__class__
        self._descriptors: tp.Dict[str, tp.Any] = {"function_class": self.__class__.__name__}
        self._parametrization: p.Parameter
        self.parametrization = parametrization
        self._function = function
        # if this is not a function bound to this very instance, add the function/callable name to the descriptors
        if not hasattr(function, '__self__') or function.__self__ != self:  # type: ignore
            name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
            self._descriptors.update(name=name)
        if hasattr(self, "get_postponing_delay"):
            raise RuntimeError('"get_posponing_delay" has been replaced by "compute_pseudotime" and has been  aggressively deprecated')
        if hasattr(self, "noisefree_function"):
            raise RuntimeError('"noisefree_function" has been replaced by "evaluation_function" and has been  aggressively deprecated')

    def register_initialization(self, **kwargs: tp.Any) -> None:
        self._initialization_kwargs = kwargs

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

    @property
    def function(self) -> tp.Callable[..., float]:
        return self._function

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        """Call the function directly (equivaluent to parametrized_function.function(*args, **kwargs))
        """
        return self._function(*args, **kwargs)

    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
            noise_level, transform and dimension
        """
        desc = dict(self._descriptors)  # Avoid external modification
        desc.update(parametrization=self.parametrization.name, dimension=self.dimension)
        return desc

    def add_descriptors(self, **kwargs) -> None:
        self._descriptors.update(kwargs)
        
    def __repr__(self) -> str:
        """Shows the function name and its summary
        """
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
        return bool(self._descriptors == other._descriptors) and self.parametrization.name == other.parametrization.name

    def copy(self: EF) -> EF:
        """Provides a new equivalent instance of the class, possibly with
        different random initialization, to provide different equivalent test cases
        when using different seeds.
        """
        if self.__class__ != ExperimentFunction:
            if self._initialization_kwargs is None:
                raise ExperimentFunctionCopyError("Copy must be specifically implemented for each subclass of ExperimentFunction "
                                                  "(and make sure you don't use the same parametrization in the process), or "
                                                  "initialization parameters should be registered through 'register_initialization'")
            kwargs = {x: y.copy() if isinstance(y, p.Parameter) else y for x, y in self._initialization_kwargs.items()}
            output = self._initialization_func(**kwargs)
            if output.parametrization.name != self.parametrization.name:
                output.parametrization = self.parametrization.copy()
            if not output.equivalent_to(self):
                raise ExperimentFunctionCopyError(f"Copy of {self} with descriptors {self._descriptors} returned non-equivalent\n"
                                                  f"{output} with descriptors {output._descriptors}.")
        else:
            # back to standard ExperimentFunction
            output = self.__class__(self.function, self.parametrization.copy())
            output._descriptors = self.descriptors
        output.parametrization._constraint_checkers = self.parametrization._constraint_checkers
        return output

    def compute_pseudotime(self, input_parameter: tp.Any, value: float) -> float:  # pylint: disable=unused-argument
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
        return 1.

    def evaluation_function(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        """Provides a (usually "noisefree") function used at final test/evaluation time in benchmarks.

        Parameters
        ----------
        *args, **kwargs
            same as the actual function
        """
        return self.function(*args, **kwargs)


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
                print(f"New best value for {identifier}: {loss}\nwith: {string}")
    except Exception:  # pylint: disable=broad-except
        pass  # better avoir bugs for this
