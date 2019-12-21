# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from nevergrad.parametrization.parameter import Instrumentation as Instrumentation
from nevergrad.parametrization.parameter import Parameter
from nevergrad.common.typetools import ArrayLike


class InstrumentedFunction:
    """Converts a multi-argument function into a single-argument multidimensional continuous function
    which can be optimized.

    Parameters
    ----------
    function: callable
        the callable to convert
    *args, **kwargs: Any
        Any argument. Arguments of type Variable (see notes) will be instrumented,
        and others will be kept constant.

    Notes
    -----
    - Variable classes are:
        - `SoftmaxCategorical(items)`: converts a list of `n` (unordered) categorial items into an `n`-dimensional space. The returned
           element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic
           and makes the function evaluation noisy.
        - `OrderedDiscrete(items)`: converts a list of (ordered) discrete items into a 1-dimensional variable. The returned value will
           depend on the value on this dimension: low values corresponding to first elements of the list, and high values to the last.
        - `Gaussian(mean, std)`: normalizes a `n`-dimensional variable with independent Gaussian priors (1-dimension per value).
        - `Array(dim1, ...)`: casts the data from the optimization space into a np.ndarray of any shape,
          to which some transforms can be applied (see `asscalar`, `affined`, `exponentiated`, `bounded`).
          This is therefore a very flexible type of variable.
        - `Scalar(dtype)`: casts the data from the optimization space into a float or an int. It is equivalent to `Array(1).asscalar(dtype)`
          and all `Array` methods are therefore available
    - This function can then be directly used in benchmarks *if it returns a float*.
    - You can update the "_descriptors" dict attribute so that function parameterization is recorded during benchmark
    """

    def __init__(self, function: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        assert callable(function)
        self._descriptors: tp.Dict[str, tp.Any] = {"function_class": self.__class__.__name__}
        self._instrumentation = Instrumentation()  # dummy
        self.instrumentation = Instrumentation(*args, **kwargs)  # sets descriptors
        self.function = function
        self.last_call_args: tp.Optional[tp.Tuple[tp.Any, ...]] = None
        self.last_call_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None
        # if this is not a function bound to this very instance, add the function/callable name to the descriptors
        if not hasattr(function, '__self__') or function.__self__ != self:  # type: ignore
            name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
            self._descriptors.update(name=name)

    @property
    def instrumentation(self) -> Instrumentation:
        return self._instrumentation

    @instrumentation.setter
    def instrumentation(self, instrum: Instrumentation) -> None:
        self._instrumentation = instrum
        self._descriptors.update(instrumentation=instrum.name, dimension=instrum.dimension)

    @property
    def dimension(self) -> int:
        return self.instrumentation.dimension

    def data_to_arguments(self, data: ArrayLike, deterministic: bool = True) -> tp.Tuple[tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any]]:
        """Get the arguments and keyword arguments corresponding to the data

        Parameters
        ----------
        data: np.ndarray
            input data
        deterministic: bool
            whether to process the data deterministically (some Variables such as SoftmaxCategorical are stochastic).
            If True, the output is the most likely output.
        """
        return self.instrumentation.data_to_arguments(data, deterministic=deterministic)

    def arguments_to_data(self, *args: tp.Any, **kwargs: tp.Any) -> ArrayLike:
        return self.instrumentation.arguments_to_data(*args, **kwargs)

    def __call__(self, x: ArrayLike) -> tp.Any:
        self.last_call_args, self.last_call_kwargs = self.data_to_arguments(x, deterministic=False)
        return self.function(*self.last_call_args, **self.last_call_kwargs)

    def get_summary(self, data: ArrayLike) -> tp.Any:  # probably impractical for large arrays
        """Provides the summary corresponding to the provided data
        """
        return self.instrumentation.get_summary(data)

    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
            noise_level, transform and dimension
        """
        return dict(self._descriptors)  # Avoid external modification

    def __repr__(self) -> str:
        """Shows the function name and its summary
        """
        params = [f"{x}={repr(y)}" for x, y in sorted(self._descriptors.items())]
        return "Instance of {}({})".format(self.__class__.__name__, ", ".join(params))

    def __eq__(self, other: tp.Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
        if other.__class__ != self.__class__:
            return False
        return bool(self._descriptors == other._descriptors)


X = tp.TypeVar("X")


# The following class is a huge (breaking) simplification of InstrumentedFunction
# Which will need to be propagated to the benchmarks

class ParametrizedFunction(tp.Generic[X]):
    """Combines a function and its parametrization

    Parameters
    ----------
    function: callable
        the callable to convert
    parameter: Parameter
        the parametrization of the function

    Notes
    -----
    - This function can then be directly used in benchmarks *if it returns a float*.
    - You can update the "_descriptors" dict attribute so that function parameterization is recorded during benchmark
    """

    def __init__(self, function: tp.Callable[..., X], parameter: Parameter) -> None:
        assert callable(function)
        self._descriptors: tp.Dict[str, tp.Any] = {"function_class": self.__class__.__name__}
        self._parameter = parameter
        self.parameter = parameter
        self._function = function
        # if this is not a function bound to this very instance, add the function/callable name to the descriptors
        if not hasattr(function, '__self__') or function.__self__ != self:  # type: ignore
            name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
            self._descriptors.update(name=name)

    @property
    def parameter(self) -> Parameter:
        return self._parameter

    @parameter.setter
    def parameter(self, parameter: Parameter) -> None:
        self._parameter = parameter
        self._parameter.freeze()
        self._descriptors.update(parameter=parameter.name, dimension=parameter.dimension)

    @property
    def function(self) -> tp.Callable[..., X]:
        return self._function

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> X:
        """Call the function directly
        """
        return self._function(*args, **kwargs)

    @property
    def descriptors(self) -> tp.Dict[str, tp.Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
            noise_level, transform and dimension
        """
        return dict(self._descriptors)  # Avoid external modification

    def __repr__(self) -> str:
        """Shows the function name and its summary
        """
        params = [f"{x}={repr(y)}" for x, y in sorted(self._descriptors.items())]
        return "Instance of {}({})".format(self.__class__.__name__, ", ".join(params))

    def is_equivalent(self, other: tp.Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal
        (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
        if other.__class__ != self.__class__:
            return False
        return bool(self._descriptors == other._descriptors)
