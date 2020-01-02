# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Any, Tuple, Dict, Optional, Callable
import numpy as np
from ..common.typetools import ArrayLike
from .variables import _Constant, wrap_arg
from .core import Variable, ArgsKwargs


class NestedVariables(Variable):
    """Variable nesting subvariables results

    Parameters
    ----------
    *args:
        Any positional argument. Arguments of type Variable (see note) will serve for instrumentation
        and others will be kept constant.
    **kwargs
        Any keyword argument. Arguments of type Variable (see note) will serve for instrumentation
        and others will be kept constant.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.keywords: Tuple[Optional[str], ...] = ()
        self.variables: List[Variable] = []
        self._set_args_kwargs(args, kwargs)

    @property
    def args(self) -> Tuple[Variable, ...]:
        """List of variables passed as positional arguments
        """
        return tuple(arg for name, arg in zip(self.keywords, self.variables) if name is None)

    @property
    def kwargs(self) -> Dict[str, Variable]:
        """Dictionary of variables passed as named arguments
        """
        return {name: arg for name, arg in zip(self.keywords, self.variables) if name is not None}

    def _set_args_kwargs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        self.keywords, arguments = self._make_argument_keywords_and_list(args, kwargs)
        self.variables = [_Constant.convert_non_instrument(a) for a in arguments]
        num_instru = len(set(id(i) for i in self.variables))
        assert len(self.variables) == num_instru, "All instruments must be different (sharing is not supported)"
        #
        params = [y.name for x, y in zip(self.keywords, self.variables) if x is None]
        params += [f"{x}={y.name}" for x, y in zip(self.keywords, self.variables) if x is not None]
        name = "NV({})".format(",".join(params))
        self._specs.update(
            dimension=sum(i.dimension for i in self.variables),
            continuous=all(v.continuous for v in self.variables),
            noisy=any(v.noisy for v in self.variables),
            nargs=len(args),
            kwargs_keys=set(kwargs.keys()),
            name=name
        )

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        super()._set_random_state(random_state)
        assert self._random_state is not None
        if self.variables:
            for var in self.variables:
                var._random_state = self._random_state

    @staticmethod
    def _make_argument_keywords_and_list(
            args: Tuple[Any, ...],
            kwargs: Dict[str, Any]
    ) -> Tuple[Tuple[Optional[str], ...], Tuple[Any, ...]]:
        """Converts *args and **kwargs to a tuple of keywords (with None for positional),
        and the corresponding tuple of values.

        Eg:
        _make_argument_keywords_and_list(3, z="blublu", machin="truc")
        >>> (None, "machin", "z"), (3, "truc", "blublu")
        """
        keywords: Tuple[Optional[str], ...] = tuple([None] * len(args) + sorted(kwargs))  # type: ignore
        arguments: Tuple[Any, ...] = args + tuple(kwargs[x] for x in keywords if x is not None)
        return keywords, arguments

    def _split_data(self, data: np.ndarray) -> List[np.ndarray]:
        """Splits the input data in chunks corresponding to each of the variables in self.variables

        Parameters
        ----------
        data: ArrayLike (list/tuple of floats, np.ndarray)
            the data in the optimization space

        Returns
        -------
        List[np.ndarray]
            the list of data chunks corresponding to each variable in self.variables
        """
        # this functions should be tested
        data = data.ravel()
        data_parts: List[np.ndarray] = []
        start, end = 0, 0
        for variable in self.variables:
            end = start + variable.dimension
            data_parts.append(data[start: end])
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"
        return data_parts

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        data_parts = self._split_data(data)
        outputs = [var.data_to_arguments(d, deterministic=deterministic) for var, d in zip(self.variables, data_parts)]
        args = tuple(arg for name, arg in zip(self.keywords, outputs) if name is None)
        kwargs = {name: arg for name, arg in zip(self.keywords, outputs) if name is not None}
        return args, kwargs

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        keywords, arguments = self._make_argument_keywords_and_list(args, kwargs)
        assert keywords == self.keywords, (f"Passed argument pattern (positional Vs named) was:\n{keywords}\n"
                                           f"but expected:\n{self.keywords}")
        data = list(itertools.chain.from_iterable([var.arguments_to_data(*vargs, **vkwargs)
                                                   for var, (vargs, vkwargs) in zip(self.variables, arguments)]))
        return np.array(data)

    def instrument(self, function: Callable[..., Any]) -> "InstrumentedFunction":
        return InstrumentedFunction(function, *self.args, **self.kwargs)

    def __repr__(self) -> str:
        ivars = [x.name if isinstance(x, _Constant) else x for x in self.variables]  # hide constants
        params = [f"{y}" for x, y in zip(self.keywords, ivars) if x is None]
        params += [f"{x}={y}" for x, y in zip(self.keywords, ivars) if x is not None]
        return "{}({})".format(self.__class__.__name__, ", ".join(params))

    def get_summary(self, data: ArrayLike) -> Any:
        """Provides the summary string corresponding to the provided data

        Note
        ----
        This is impractical for large arrays
        """
        strings = []
        splitted_data = self._split_data(np.asarray(data))
        for k, (keyword, var, d) in enumerate(zip(self.keywords, self.variables, splitted_data)):
            if not isinstance(var, _Constant):
                explanation = var.get_summary(d)
                sname = f"arg #{k + 1}" if keyword is None else f'kwarg "{keyword}"'
                strings.append(f"{sname}: {explanation}")
        return " - " + "\n - ".join(strings)


class Instrumentation(NestedVariables):
    """Class handling arguments instrumentation, and providing conversion to and from data.

    Parameters
    ----------
    *args:
        Any positional argument. Arguments of type Variable (see note) will serve for instrumentation
        and others will be kept constant.
    **kwargs
        Any keyword argument. Arguments of type Variable (see note) will serve for instrumentation
        and others will be kept constant.

    Note
    ----
    * Core Variable classes are:

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
        - `Log(a_min, a_max)`: for log distributed data between two bounds. Under the hood this uses an `Scalar` with an
          appropriate set of transforms (including clipping for the bounds).
    * Depending on the variables, `Instrumentation` can be noisy (`SoftmaxCategorical` in non-deterministic mode), and not continuous
      (`SoftmaxCategorical` in deterministic mode, `OrderedDiscrete`, `Array` with int casting). Some optimizers may not be able
      to deal with these cases properly.
    * Since instrumentation can have a stochastic behavior, it is responsible of holding the random state which optimizers
      pull from. This random state is initialized lazily, and can be modified
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.keywords: Tuple[Optional[str], ...] = ()
        self.probably_noisy = False  # True if for some reason we conjecture that the objective function is noisy.
        self.is_nonmetrizable = False  # True if there is no metric on the domain.
        self.variables: List[Variable] = []
        self._set_args_kwargs(args, kwargs)

    def _set_args_kwargs(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
        super()._set_args_kwargs(args, kwargs)
        assert all(v.nargs == 1 and not v.kwargs_keys for v in self.variables), "Not yet supported"
        assert self.name.startswith("NV(")
        self._specs.update(
            nargs=len(args),
            kwargs_keys=set(kwargs.keys()),
            name=self.name[3:-1]
        )

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        nested_args, nested_kwargs = super()._data_to_arguments(data, deterministic=deterministic)
        return tuple(a[0][0] for a in nested_args), {key: a[0][0] for key, a in nested_kwargs.items()}

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return super()._arguments_to_data(*(wrap_arg(x) for x in args), **{k: wrap_arg(x) for k, x in kwargs.items()})


class InstrumentedFunction:
    """InstrumentedFunction is being aggressively deprecated. Conversion depends on your use case:
    - for optimization purpose: directly provide ng.Instrumentation to the optimizer, it will
       provide candidates with fields 'args' and 'kwargs' that match the instrumentation.
    - for benchmarks: derive from ng.functions.ExperimentFunction. Main differences are:
       calls to __call__ directly forwards the main function (instead of converting from data space),
       __init__ takes exactly two arguments (main function and parametrization/instrumentation) and
      instrumentation is renamed to parametrization for forward compatibility.
    """

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(self.__doc__)
