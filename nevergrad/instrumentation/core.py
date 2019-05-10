# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import itertools
from typing import List, Any, Tuple, Dict, Optional, Callable
import numpy as np
from ..common.typetools import ArrayLike
from . import utils
from . import variables


class Instrumentation:
    """Class handling arguments instrumentation, and providing conversion to and from data.

    Parameters
    ----------
    *args, **kwargs: Any
        Any argument. Arguments of type Variable (see note) will serve for instrumentation
        and others will be kept constant.

    Note
    ----
    * Variable classes are:
      - `SoftmaxCategorical`: converts a list of `n` (unordered) categorial variables into an `n`-dimensional space. The returned
         element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic
         and makes the function evaluation noisy.
      - `OrderedDiscrete`: converts a list of (ordered) discrete variables into a 1-dimensional variable. The returned value will
         depend on the value on this dimension: low values corresponding to first elements of the list, and high values to the last.
      - `Gaussian`: normalizes a `n`-dimensional variable with independent Gaussian priors (1-dimension per value).
      - `Array`: casts the data from the optimization space into a np.ndarray of any shape, to which some transforms can be applied
        (see `asscalar`, `affined`, `exponentiated`, `bounded`). This makes it a very flexible type of variable.
    * Depending on the variables, Instrumentation can be noisy (SoftmaxCategorical in non-deterministic mode), and not continuous
      (SoftmaxCategorical in deterministic mode, `OrderedDiscrete`, `Array` with int casting). Some optimizers may not be able
      to deal with these cases properly.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.names: Tuple[Optional[str], ...] = ()
        self.variables: List[utils.Variable[Any]] = []
        self._set_args_kwargs(args, kwargs)
        self._name: Optional[str] = None

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return format(self, "short")

    @property
    def continuous(self) -> bool:
        """Wether the instrumentation is continuous, i.e. all underlying variables are continuous.
        """
        return all(v.continuous for v in self.variables)

    @property
    def noisy(self) -> bool:
        """Wether the instrumentation is noisy, i.e. at least one of the underlying variable is noisy.
        """
        return any(v.noisy for v in self.variables)

    def with_name(self, name: str) -> "Instrumentation":
        """Sets a name and return the current instrumentation (for chaining)
        """
        self._name = name
        return self

    def _set_args_kwargs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.names, arguments = self._make_argument_names_and_list(args, kwargs)
        self.variables: List[utils.Variable[Any]] = [variables._Constant.convert_non_instrument(a) for a in arguments]
        num_instru = len(set(id(i) for i in self.variables))
        assert len(self.variables) == num_instru, "All instruments must be different (sharing is not supported)"

    @property
    def dimension(self) -> int:
        return sum(i.dimension for i in self.variables)

    @property
    def args(self) -> Tuple[utils.Variable[Any], ...]:
        """List of variables passed as positional arguments
        """
        return tuple(arg for name, arg in zip(self.names, self.variables) if name is None)

    @property
    def kwargs(self) -> Dict[str, utils.Variable[Any]]:
        """Dictionary of variables passed as named arguments
        """
        return {name: arg for name, arg in zip(self.names, self.variables) if name is not None}

    @staticmethod
    def _make_argument_names_and_list(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[Tuple[Optional[str], ...], Tuple[Any, ...]]:
        """Converts *args and **kwargs to a tuple of names (with None for positional),
        and the corresponding tuple of values.

        Eg:
        _make_argument_names_and_list(3, z="blublu", machin="truc")
        >>> (None, "machin", "z"), (3, "truc", "blublu")
        """
        names: Tuple[Optional[str], ...] = tuple([None] * len(args) + sorted(kwargs))  # type: ignore
        arguments: Tuple[Any, ...] = args + tuple(kwargs[x] for x in names if x is not None)
        return names, arguments

    def data_to_arguments(self, data: ArrayLike, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Converts data to arguments
        """
        arguments = utils.process_variables(self.variables, data, deterministic=deterministic)
        args = tuple(arg for name, arg in zip(self.names, arguments) if name is None)
        kwargs = {name: arg for name, arg in zip(self.names, arguments) if name is not None}
        return args, kwargs

    def arguments_to_data(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """Converts arguments to data

        Note
        ----
        - you need to input the arguments in the same way than at initialization
          with regard to positional and named arguments.
        - this process is simplified, and is deterministic. Depending on your instrumentation,
          you will probably not recover the same data.
        """
        names, arguments = self._make_argument_names_and_list(args, kwargs)
        assert names == self.names, (f"Passed argument pattern (positional Vs named) was:\n{names}\n"
                                     f"but expected:\n{self.names}")
        data = list(itertools.chain.from_iterable([instrument.argument_to_data(arg)
                                                   for instrument, arg in zip(self.variables, arguments)]))
        return np.array(data)

    def instrument(self, function: Callable[..., Any]) -> "InstrumentedFunction":
        return InstrumentedFunction(function, *self.args, **self.kwargs)

    def __format__(self, format_spec: str) -> str:
        arguments = [format(x, format_spec) for x in self.args]
        sorted_kwargs = [(name, format(self.kwargs[name], format_spec)) for name in sorted(self.kwargs)]
        all_params = arguments + [f"{x}={y}" for x, y in sorted_kwargs]
        if format_spec == "short":
            return ",".join(all_params)
        return "{}({})".format(self.__class__.__name__, ", ".join(all_params))

    def __repr__(self) -> str:
        return f"{self:display}"

    def get_summary(self, data: ArrayLike) -> Any:
        """Provides the summary string corresponding to the provided data

        Note
        ----
        This is impractical for large arrays
        """
        strings = []
        splitted_data = utils.split_data(data, self.variables)
        for k, (name, var, d) in enumerate(zip(self.names, self.variables, splitted_data)):
            if not isinstance(var, variables._Constant):
                explanation = var.get_summary(d)
                sname = f"arg #{k + 1}" if name is None else f'kwarg "{name}"'
                strings.append(f"{sname}: {explanation}")
        return " - " + "\n - ".join(strings)


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
        - `SoftmaxCategorical`: converts a list of `n` (unordered) categorial variables into an `n`-dimensional space. The returned
           element will be sampled as the softmax of the values on these dimensions. Be cautious: this process is non-deterministic
           and makes the function evaluation noisy.
        - `OrderedDiscrete`: converts a list of (ordered) discrete variables into a 1-dimensional variable. The returned value will
           depend on the value on this dimension: low values corresponding to first elements of the list, and high values to the last.
        - `Gaussian`: normalizes a `n`-dimensional variable with independent Gaussian priors (1-dimension per value).
        - `Array`: casts the data from the optimization space into a np.ndarray of any shape, to which some transforms can be applied
          (see `asscalar`, `affined`, `exponentiated`, `bounded`). This makes it a very flexible type of variable.
    - This function can then be directly used in benchmarks *if it returns a float*.
    - You can update the "_descriptors" dict attribute so that function parameterization is recorded during benchmark
    """

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        assert callable(function)
        self._descriptors: Dict[str, Any] = {"function_class": self.__class__.__name__}
        self._instrumentation = Instrumentation()  # dummy
        self.instrumentation = Instrumentation(*args, **kwargs)  # sets descriptors
        self.function = function
        self.last_call_args: Optional[Tuple[Any, ...]] = None
        self.last_call_kwargs: Optional[Dict[str, Any]] = None
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

    def data_to_arguments(self, data: ArrayLike, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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

    def arguments_to_data(self, *args: Any, **kwargs: Any) -> ArrayLike:
        return self.instrumentation.arguments_to_data(*args, **kwargs)

    def __call__(self, x: ArrayLike) -> Any:
        self.last_call_args, self.last_call_kwargs = self.data_to_arguments(x, deterministic=False)
        return self.function(*self.last_call_args, **self.last_call_kwargs)

    def get_summary(self, data: ArrayLike) -> Any:  # probably impractical for large arrays
        """Provides the summary corresponding to the provided data
        """
        return self.instrumentation.get_summary(data)

    def convert_to_arguments(self, data: ArrayLike, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        warnings.warn("convert_to_arguments is deprecated, please use data_to_arguments instead")
        return self.data_to_arguments(data, deterministic=deterministic)

    def convert_to_data(self, *args: Any, **kwargs: Any) -> ArrayLike:
        warnings.warn("convert_to_data is deprecated, please use arguments_to_data instead")
        return self.arguments_to_data(*args, **kwargs)

    @property
    def descriptors(self) -> Dict[str, Any]:
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
            noise_level, transform and dimension
        """
        return dict(self._descriptors)  # Avoid external modification

    def __repr__(self) -> str:
        """Shows the function name and its summary
        """
        params = [f"{x}={repr(y)}" for x, y in sorted(self._descriptors.items())]
        return "Instance of {}({})".format(self.__class__.__name__, ", ".join(params))

    def __eq__(self, other: Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
        if other.__class__ != self.__class__:
            return False
        return bool(self._descriptors == other._descriptors)
