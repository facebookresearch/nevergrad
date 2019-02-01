# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import itertools
from typing import List, Any, Tuple, Dict, Optional, Callable
import numpy as np
from ..common.typetools import ArrayLike
from ..functions import base
from . import utils
from . import variables


class Instrumentation:
    """Class handling arguments instrumentation, and providing conversion to and from data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.names: Tuple[Optional[str], ...] = ()
        self.instruments: List[utils.Variable] = []
        self._set_args_kwargs(args, kwargs)
        self._name: Optional[str] = None

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        return format(self, "short")

    def with_name(self, name: str) -> "Instrumentation":
        """Sets a name and return the current instrumentation (for chaining)
        """
        self._name = name
        return self

    def _set_args_kwargs(self, args: Tuple[Any, ...], kwargs: Dict) -> None:
        self.names, arguments = self._make_argument_names_and_list(args, kwargs)
        self.instruments: List[utils.Variable] = [variables._Constant.convert_non_instrument(a) for a in arguments]
        num_instru = len(set(id(i) for i in self.instruments))
        assert len(self.instruments) == num_instru, "All instruments must be different (sharing is not supported)"

    @property
    def dimension(self) -> int:
        return sum(i.dimension for i in self.instruments)

    @property
    def args(self) -> Tuple[utils.Variable, ...]:
        """List of instruments passed as positional arguments
        """
        return tuple(arg for name, arg in zip(self.names, self.instruments) if name is None)

    @property
    def kwargs(self) -> Dict[str, utils.Variable]:
        """Dictionary of instruments passed as named arguments
        """
        return {name: arg for name, arg in zip(self.names, self.instruments) if name is not None}

    @staticmethod
    def _make_argument_names_and_list(args: Tuple[Any, ...], kwargs: Dict) -> Tuple[Tuple[Optional[str], ...], Tuple[Any, ...]]:
        """Converts *args and **kwargs to a tuple of names (with None for positional),
        and the corresponding tuple of values.

        Eg:
        _make_argument_names_and_list(3, z="blublu", machin="truc")
        >>> (None, "machin", "z"), (3, "truc", "blublu")
        """
        names: Tuple[Optional[str], ...] = tuple([None] * len(args) + sorted(kwargs))
        arguments: Tuple[Any, ...] = args + tuple(kwargs[x] for x in names if x is not None)
        return names, arguments

    def data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Converts data to arguments
        """
        arguments = utils.process_instruments(self.instruments, data, deterministic=deterministic)
        args = tuple(arg for name, arg in zip(self.names, arguments) if name is None)
        kwargs = {name: arg for name, arg in zip(self.names, arguments) if name is not None}
        return args, kwargs

    def arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
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
        data = list(itertools.chain.from_iterable([instrument.process_arg(arg) for instrument, arg in zip(self.instruments, arguments)]))
        return np.array(data)

    def instrument(self, function: Callable) -> "InstrumentedFunction":
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

    def get_summary(self, data: np.ndarray) -> Any:
        """Provides the summary string corresponding to the provided data

        Note
        ----
        This is impractical for large arrays
        """
        strings = []
        splitted_data = utils.split_data(data, self.instruments)
        for k, (name, var, d) in enumerate(zip(self.names, self.instruments, splitted_data)):
            if not isinstance(var, variables._Constant):
                explanation = var.get_summary(d)
                sname = f"arg #{k + 1}" if name is None else f'kwarg "{name}"'
                strings.append(f"{sname}: {explanation}")
        return " - " + "\n - ".join(strings)


class InstrumentedFunction(base.BaseFunction):
    """Converts a multi-argument function into a mono-argument multidimensional continuous function
    which can be optimized.

    Parameters
    ----------
    function: callable
        the callable to convert
    *args, **kwargs: Any
        Any argument. Arguments of type variables.SoftmaxCategorical or variables.Gaussian will be instrumented
        and others will be kept constant.

    Note
    ----
    - Tokens can be:
      - DiscreteToken(list_of_n_possible_values): converted into a n-dim array, corresponding to proba for each value
      - GaussianToken(mean, std, shape=None): a Gaussian variable (shape=None) or array.
    - This function can then be directly used in benchmarks *if it returns a float*.

    """

    def __init__(self, function: Callable, *args: Any, **kwargs: Any) -> None:
        assert callable(function)
        self.instrumentation = Instrumentation(*args, **kwargs)
        super().__init__(dimension=self.instrumentation.dimension)
        # keep track of what is instrumented (but "how" is probably too long/complex)
        name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
        self._descriptors.update(name=name, instrumentation=self.instrumentation.name)
        self._function = function
        self.last_call_args: Optional[Tuple[Any, ...]] = None
        self.last_call_kwargs: Optional[Dict[str, Any]] = None

    def data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
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

    def oracle_call(self, x: np.ndarray) -> Any:
        self.last_call_args, self.last_call_kwargs = self.data_to_arguments(x, deterministic=False)
        return self._function(*self.last_call_args, **self.last_call_kwargs)

    def __call__(self, x: np.ndarray) -> Any:
        # BaseFunction __call__ method should generally not be overriden,
        # but here that would mess up with typing, and I would rather not constrain
        # user to return only floats.
        x = self.transform(x)
        return self.oracle_call(x)

    def get_summary(self, data: np.ndarray) -> Any:  # probably impractical for large arrays
        """Provides the summary corresponding to the provided data
        """
        return self.instrumentation.get_summary(data)

    def convert_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        warnings.warn("convert_to_arguments is deprecated, please use data_to_arguments instead")
        return self.data_to_arguments(data, deterministic=deterministic)

    def convert_to_data(self, *args: Any, **kwargs: Any) -> ArrayLike:
        warnings.warn("convert_to_data is deprecated, please use arguments_to_data instead")
        return self.arguments_to_data(*args, **kwargs)
