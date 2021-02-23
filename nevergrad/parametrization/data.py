# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.(an
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import warnings
import numpy as np
import nevergrad.common.typing as tp

from nevergrad.common import errors
from . import _layering
from . import core
from .container import Dict
from . import utils
from . import transforms as trans


# pylint: disable=no-value-for-parameter


D = tp.TypeVar("D", bound="Data")
P = tp.TypeVar("P", bound=core.Parameter)
# L = tp.TypeVar("L", bound=_layering.Layered)
BL = tp.TypeVar("BL", bound="BoundLayer")


def _param_string(parameters: Dict) -> str:
    """Hacky helper for nice-visualizatioon"""
    substr = f"[{parameters._get_parameters_str()}]"
    if substr == "[]":
        substr = ""
    return substr


class Mutation(core.Parameter):
    """Custom mutation or recombination
    This is an experimental API

    Either implement:
    - `_apply_array`  which provides a new np.ndarray from a list of arrays
    - `apply` which updates the first p.Array instance

    Mutation should take only one p.Array instance as argument, while
    Recombinations should take several
    """

    # NOTE: this API should disappear in favor of the layer API
    # (a layer can modify the mutation scheme)

    # pylint: disable=unused-argument
    value: core.ValueProperty[tp.Callable[[tp.Sequence[D]], None]] = core.ValueProperty()

    def __init__(self, **kwargs: tp.Any) -> None:
        super().__init__()
        self.parameters = Dict(**kwargs)

    def _layered_get_value(self) -> tp.Callable[[tp.Sequence[D]], None]:
        return self.apply

    def _layered_set_value(self, value: tp.Any) -> None:
        raise RuntimeError("Mutation cannot be set.")

    def _get_name(self) -> str:
        return super()._get_name() + _param_string(self.parameters)

    def apply(self, arrays: tp.Sequence[D]) -> None:
        new_value = self._apply_array([a._value for a in arrays])
        arrays[0]._value = new_value

    def _apply_array(self, arrays: tp.Sequence[np.ndarray]) -> np.ndarray:
        raise RuntimeError("Mutation._apply_array should either be implementer or bypassed in Mutation.apply")
        return np.array([])  # pylint: disable=unreachable

    def get_standardized_data(  # pylint: disable=unused-argument
        self: P, *, reference: tp.Optional[P] = None
    ) -> np.ndarray:
        return np.array([])


# pylint: disable=too-many-arguments, too-many-instance-attributes,abstract-method
class Data(core.Parameter):
    """Array parameter with customizable mutation and recombination.

    Parameters
    ----------
    init: np.ndarray, or None
        initial value of the array (defaults to 0, with a provided shape)
    shape: tuple of ints, or None
        shape of the array, to be provided iff init is not provided
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    More specific behaviors can be obtained throught the following methods:
    set_bounds, set_mutation, set_integer_casting
    """

    def __init__(
        self,
        *,
        init: tp.Optional[tp.ArrayLike] = None,
        shape: tp.Optional[tp.Tuple[int, ...]] = None,
        mutable_sigma: bool = False,
    ) -> None:
        sigma = Log(init=1.0, exponent=2.0, mutable_sigma=False) if mutable_sigma else 1.0
        super().__init__()
        self.parameters = Dict(sigma=sigma, recombination="average", mutation="gaussian")
        err_msg = 'Exactly one of "init" or "shape" must be provided'
        self.parameters._ignore_in_repr = dict(sigma="1.0", recombination="average", mutation="gaussian")
        if init is not None:
            if shape is not None:
                raise ValueError(err_msg)
            self._value = np.array(init, copy=False)
        elif shape is not None:
            assert isinstance(shape, tuple) and all(
                isinstance(n, int) for n in shape
            ), f"Shape incorrect: {shape}."
            self._value = np.zeros(shape)
        else:
            raise ValueError(err_msg)
        self.exponent: tp.Optional[float] = None
        self.bounds: tp.Tuple[tp.Optional[np.ndarray], tp.Optional[np.ndarray]] = (None, None)
        self.bound_transform: tp.Optional[trans.BoundTransform] = None
        self.full_range_sampling = False
        self._ref_data: tp.Optional[np.ndarray] = None
        self.add_layer(_layering.ArrayCasting())

    @property
    def dimension(self) -> int:
        return int(np.prod(self._value.shape))

    def _compute_descriptors(self) -> utils.Descriptors:
        return utils.Descriptors(continuous=not self.integer)

    def _get_name(self) -> str:
        cls = self.__class__.__name__
        descriptors: tp.List[str] = (
            ["int"]
            if self.integer
            else ([str(self._value.shape).replace(" ", "")] if self._value.shape != (1,) else [])
        )
        descriptors += [f"exp={self.exponent}"] if self.exponent is not None else []
        descriptors += [f"{self.bound_transform.name}"] if self.bound_transform is not None else []
        descriptors += ["constr"] if self._constraint_checkers else []
        description = ""
        if descriptors:
            description = "{{{}}}".format(",".join(descriptors))
        return f"{cls}{description}" + _param_string(self.parameters)

    @property
    def sigma(self) -> tp.Union["Array", "Scalar"]:
        """Value for the standard deviation used to mutate the parameter"""
        return self.parameters["sigma"]  # type: ignore

    def _layered_sample(self: D) -> D:
        if not self.full_range_sampling:
            child = self.spawn_child()
            child.mutate()
            return child
        child = self.spawn_child()
        func = (lambda x: x) if self.exponent is None else self._to_reduced_space  # noqa
        std_bounds = tuple(func(b * np.ones(self._value.shape)) for b in self.bounds)
        diff = std_bounds[1] - std_bounds[0]
        new_data = std_bounds[0] + self.random_state.uniform(0, 1, size=diff.shape) * diff
        if self.exponent is None:
            new_data = self._to_reduced_space(new_data)
        child.set_standardized_data(new_data - self._get_ref_data(), deterministic=False)
        child.heritage["lineage"] = child.uid
        return child

    # pylint: disable=unused-argument
    def set_bounds(
        self: D,
        lower: tp.BoundValue = None,
        upper: tp.BoundValue = None,
        method: str = "bouncing",
        full_range_sampling: tp.Optional[bool] = None,
    ) -> D:
        """Bounds all real values into [lower, upper] using a provided method

        Parameters
        ----------
        lower: float or None
            minimum value
        upper: float or None
            maximum value
        method: str
            One of the following choices:

            - "bouncing": bounce on border (at most once). This is a variant of clipping,
               avoiding bounds over-samping (default).
            - "clipping": clips the values inside the bounds. This is efficient but leads
              to over-sampling on the bounds.
            - "constraint": adds a constraint (see register_cheap_constraint) which leads to rejecting mutations
              reaching beyond the bounds. This avoids oversampling the boundaries, but can be inefficient in large
              dimension.
            - "arctan": maps the space [lower, upper] to to all [-inf, inf] using arctan transform. This is efficient
              but it completely reshapes the space (a mutation in the center of the space will be larger than a mutation
              close to the bounds), and reaching the bounds is equivalent to reaching the infinity.
            - "tanh": same as "arctan", but with a "tanh" transform. "tanh" saturating much faster than "arctan", it can lead
              to unexpected behaviors.
        full_range_sampling: Optional bool
            Changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            or the sampling optimizers, to creating a child with a value sampled uniformly (or log-uniformly) within
            the while range of the bounds. The "sample" method is used by some algorithms to create an initial population.
            This is activated by default if both bounds are provided.

        Notes
        -----
        - "tanh" reaches the boundaries really quickly, while "arctan" is much softer
        - only "clipping" accepts partial bounds (None values)
        """  # TODO improve description of methods
        bounds = tuple(
            a if isinstance(a, np.ndarray) or a is None else np.array([a], dtype=float)
            for a in (lower, upper)
        )
        both_bounds = all(b is not None for b in bounds)
        if full_range_sampling is None:
            full_range_sampling = both_bounds
        # preliminary checks
        if self.bound_transform is not None:
            raise RuntimeError("A bounding method has already been set")
        if full_range_sampling and not both_bounds:
            raise ValueError("Cannot use full range sampling if both bounds are not set")
        checker = utils.BoundChecker(*bounds)
        if not checker(self.value):
            raise ValueError("Current value is not within bounds, please update it first")
        if not (lower is None or upper is None):
            if (bounds[0] >= bounds[1]).any():  # type: ignore
                raise ValueError(f"Lower bounds {lower} should be strictly smaller than upper bounds {upper}")
        # update instance
        transforms = dict(
            clipping=trans.Clipping,
            arctan=trans.ArctanBound,
            tanh=trans.TanhBound,
            gaussian=trans.CumulativeDensity,
        )
        transforms["bouncing"] = functools.partial(trans.Clipping, bounce=True)  # type: ignore
        if method in transforms:
            if self.exponent is not None and method not in ("clipping", "bouncing"):
                raise ValueError(f'Cannot use method "{method}" in logarithmic mode')
            self.bound_transform = transforms[method](*bounds)
        elif method == "constraint":
            self.register_cheap_constraint(checker)
        else:
            avail = ["constraint"] + list(transforms)
            raise ValueError(f"Unknown method {method}, available are: {avail}\nSee docstring for more help.")
        self.bounds = bounds  # type: ignore
        self.full_range_sampling = full_range_sampling
        # warn if sigma is too large for range
        if both_bounds and method != "tanh":  # tanh goes to infinity anyway
            std_bounds = tuple(self._to_reduced_space(b) for b in self.bounds)  # type: ignore
            min_dist = np.min(np.abs(std_bounds[0] - std_bounds[1]).ravel())
            if min_dist < 3.0:
                warnings.warn(
                    f"Bounds are {min_dist} sigma away from each other at the closest, "
                    "you should aim for at least 3 for better quality."
                )
        return self

    def set_recombination(self: D, recombination: tp.Union[None, str, core.Parameter]) -> D:
        self.parameters._content["recombination"] = (
            recombination if isinstance(recombination, core.Parameter) else core.Constant(recombination)
        )
        return self

    def mutate(self) -> None:
        """Mutate parameters of the instance, and then its value"""
        self._check_frozen()
        self._subobjects.apply("mutate")
        mutation = self.parameters["mutation"].value
        if isinstance(mutation, str):
            if mutation in ["gaussian", "cauchy"]:
                func = (
                    self.random_state.normal if mutation == "gaussian" else self.random_state.standard_cauchy
                )
                self.set_standardized_data(func(size=self.dimension), deterministic=False)
            else:
                raise NotImplementedError('Mutation "{mutation}" is not implemented')
        elif isinstance(mutation, Mutation):
            mutation.apply([self])
        elif callable(mutation):
            mutation([self])
        else:
            raise TypeError("Mutation must be a string, a callable or a Mutation instance")

    def set_mutation(
        self: D,
        sigma: tp.Optional[tp.Union[float, core.Parameter]] = None,
        exponent: tp.Optional[float] = None,
        custom: tp.Optional[tp.Union[str, core.Parameter]] = None,
    ) -> D:
        """Output will be cast to integer(s) through deterministic rounding.

        Parameters
        ----------
        sigma: Array/Log or float
            The standard deviation of the mutation. If a Parameter is provided, it will replace the current
            value. If a float is provided, it will either replace a previous float value, or update the value
            of the Parameter.
        exponent: float
            exponent for the logarithmic mode. With the default sigma=1, using exponent=2 will perform
            x2 or /2 "on average" on the value at each mutation.
        custom: str or Parameter
            custom mutation which can be a string ("gaussian" or "cauchy")
            or Mutation/Recombination like object
            or a Parameter which value is either of those

        Returns
        -------
        self
        """
        if sigma is not None:
            # just replace if an actual Parameter is provided as sigma, else update value (parametrized or not)
            if isinstance(sigma, core.Parameter) or isinstance(
                self.parameters._content["sigma"], core.Constant
            ):
                self.parameters._content["sigma"] = core.as_parameter(sigma)
            else:
                self.sigma.value = sigma  # type: ignore
        if exponent is not None:
            if self.bound_transform is not None and not isinstance(self.bound_transform, trans.Clipping):
                raise RuntimeError(
                    f"Cannot set logarithmic transform with bounding transform {self.bound_transform}, "
                    "only clipping and constraint bounding methods can accept itp."
                )
            if exponent <= 1.0:
                raise ValueError("Only exponents strictly higher than 1.0 are allowed")
            if np.min(self._value.ravel()) <= 0:
                raise RuntimeError(
                    "Cannot convert to logarithmic mode with current non-positive value, please update it firstp."
                )
            self.exponent = exponent
        if custom is not None:
            self.parameters._content["mutation"] = core.as_parameter(custom)
        return self

    def set_integer_casting(self: D) -> D:
        """Output will be cast to integer(s) through deterministic rounding.

        Returns
        -------
        self

        Note
        ----
        Using integer casting makes the parameter discrete which can make the optimization more
        difficult. It is especially ill-advised to use this with a range smaller than 10, or
        a sigma lower than 1. In those cases, you should rather use a TransitionChoice instead.
        """
        return self.add_layer(_layering.IntegerCasting())

    @property
    def integer(self) -> bool:
        return any(isinstance(x, _layering.IntegerCasting) for x in self._layers)

    # pylint: disable=unused-argument
    def _internal_set_standardized_data(
        self: D, data: np.ndarray, reference: D, deterministic: bool = False
    ) -> None:
        assert isinstance(data, np.ndarray)
        sigma = reference.sigma.value
        data_reduc = sigma * (data + reference._get_ref_data()).reshape(reference._value.shape)
        self._value = data_reduc if reference.exponent is None else reference.exponent ** data_reduc
        self._ref_data = None
        if reference.bound_transform is not None:
            self._value = reference.bound_transform.forward(self._value)

    def _internal_get_standardized_data(self: D, reference: D) -> np.ndarray:
        return reference._to_reduced_space(self._value) - reference._get_ref_data()  # type: ignore

    def _get_ref_data(self) -> np.ndarray:
        if self._ref_data is None:
            self._ref_data = self._to_reduced_space(self._value)
        return self._ref_data

    def _to_reduced_space(self, value: np.ndarray) -> np.ndarray:
        """Converts array with appropriate shapes to reduced (uncentered) space
        by applying log scaling and sigma scaling
        """
        sigma = self.sigma.value
        if self.bound_transform is not None:
            value = self.bound_transform.backward(value)
        distribval = value if self.exponent is None else np.log(value) / np.log(self.exponent)
        reduced = distribval / sigma
        return reduced.ravel()  # type: ignore

    def recombine(self: D, *others: D) -> None:
        if not others:
            return
        self._subobjects.apply("recombine", *others)
        recomb = self.parameters["recombination"].value
        if recomb is None:
            return
        all_params = [self] + list(others)
        if isinstance(recomb, str) and recomb == "average":
            all_arrays = [p.get_standardized_data(reference=self) for p in all_params]
            self.set_standardized_data(np.mean(all_arrays, axis=0), deterministic=False)
        elif isinstance(recomb, Mutation):
            recomb.apply(all_params)
        elif callable(recomb):
            recomb(all_params)
        else:
            raise ValueError(f'Unknown recombination "{recomb}"')

    def copy(self: D) -> D:
        child = super().copy()
        child._value = np.array(self._value, copy=True)
        return child

    def _layered_set_value(self, value: np.ndarray) -> None:
        self._check_frozen()
        self._ref_data = None
        if self._value.shape != value.shape:
            raise ValueError(
                f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}"
            )
        if not utils.BoundChecker(*self.bounds)(self.value):
            raise ValueError("New value does not comply with bounds")
        if self.exponent is not None and np.min(value.ravel()) <= 0:
            raise ValueError("Logirithmic values cannot be negative")
        self._value = value

    def _layered_get_value(self) -> np.ndarray:
        return self._value

    def __mod__(self: D, other: tp.Any) -> D:
        new = self.copy()
        new.add_layer(Modulo(other))
        return new


class Array(Data):

    value: core.ValueProperty[np.ndarray] = core.ValueProperty()


class Scalar(Data):
    """Parameter representing a scalar.

    Parameters
    ----------
    init: optional float
        initial value of the scalar (defaults to 0.0 if both bounds are not provided)
    lower: optional float
        minimum value if any
    upper: optional float
        maximum value if any
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Notes
    -----
    - by default, this is an unbounded scalar with Gaussian mutations.
    - if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma.
      Also, if init is not provided, it will be set to the middle value.
    - More specific behaviors can be obtained throught the following methods:
      :code:`set_bounds`, :code:`set_mutation`, :code:`set_integer_casting`
    """

    value: core.ValueProperty[float] = core.ValueProperty()

    def __init__(
        self,
        init: tp.Optional[float] = None,
        *,
        lower: tp.Optional[float] = None,
        upper: tp.Optional[float] = None,
        mutable_sigma: bool = True,
    ) -> None:
        bounded = all(a is not None for a in (lower, upper))
        no_init = init is None
        if bounded:
            if init is None:
                init = (lower + upper) / 2.0  # type: ignore
        if init is None:
            init = 0.0
        super().__init__(init=np.array([init]), mutable_sigma=mutable_sigma)
        if bounded:
            self.set_mutation(sigma=(upper - lower) / 6)  # type: ignore
        if any(a is not None for a in (lower, upper)):
            self.set_bounds(lower=lower, upper=upper, full_range_sampling=bounded and no_init)
        self.add_layer(_layering._ScalarCasting())


class Log(Scalar):
    """Parameter representing a positive variable, mutated by Gaussian mutation in log-scale.

    Parameters
    ----------
    init: float or None
        initial value of the variable. If not provided, it is set to the middle of lower and upper in log space
    exponent: float or None
        exponent for the log mutation: an exponent of 2.0 will lead to mutations by factors between around 0.5 and 2.0
        By default, it is set to either 2.0, or if the parameter is completely bounded to a factor so that bounds are
        at 3 sigma in the transformed space.
    lower: float or None
        minimum value if any (> 0)
    upper: float or None
        maximum value if any
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    This class is only a wrapper over :code:`Scalar`.
    """

    def __init__(
        self,
        *,
        init: tp.Optional[float] = None,
        exponent: tp.Optional[float] = None,
        lower: tp.Optional[float] = None,
        upper: tp.Optional[float] = None,
        mutable_sigma: bool = False,
    ) -> None:
        no_init = init is None
        bounded = all(a is not None for a in (lower, upper))
        if bounded:
            if init is None:
                init = float(np.sqrt(lower * upper))  # type: ignore
            if exponent is None:
                exponent = float(
                    np.exp((np.log(upper) - np.log(lower)) / 6.0)
                )  # 99.7% of values within the bounds
        if init is None:
            raise ValueError("You must define either a init value or both lower and upper bounds")
        if exponent is None:
            exponent = 2.0
        super().__init__(init=init, mutable_sigma=mutable_sigma)
        self.set_mutation(sigma=1.0, exponent=exponent)
        if any(a is not None for a in (lower, upper)):
            self.set_bounds(lower, upper, full_range_sampling=bounded and no_init)


# LAYERS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class BoundLayer(_layering.Layered):

    _LAYER_LEVEL = _layering.Level.OPERATION

    def __init__(
        self,
        lower: tp.BoundValue = None,
        upper: tp.BoundValue = None,
        full_range_sampling: tp.Optional[bool] = None,
    ) -> None:
        """Bounds all real values into [lower, upper]
        CAUTION: WIP

        Parameters
        ----------
        lower: float or None
            minimum value
        upper: float or None
            maximum value
        method: str
            One of the following choices:
        full_range_sampling: Optional bool
            Changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            or the sampling optimizers, to creating a child with a value sampled uniformly (or log-uniformly) within
            the while range of the bounds. The "sample" method is used by some algorithms to create an initial population.
            This is activated by default if both bounds are provided.
        """  # TODO improve description of methods
        super().__init__()
        self.bounds = tuple(
            a if isinstance(a, np.ndarray) or a is None else np.array([a], dtype=float)
            for a in (lower, upper)
        )
        both_bounds = all(b is not None for b in self.bounds)
        self.full_range_sampling = full_range_sampling
        if full_range_sampling is None:
            self.full_range_sampling = both_bounds

    def _layered_sample(self) -> "Data":
        if not self.full_range_sampling:
            return super()._layered_sample()  # type: ignore
        root = self._layers[0]
        if not isinstance(root, Data):
            raise errors.NevergradTypeError(f"BoundLayer {self} on a non-Data root {root}")
        child = root.spawn_child()
        shape = super()._layered_get_value().shape
        bounds = tuple(b * np.ones(shape) for b in self.bounds)
        new_val = root.random_state.uniform(*bounds)
        # send new val to the layer under this one for the child
        child._layers[self._index - 1]._layered_set_value(new_val)
        return child


class Modulo(BoundLayer):
    """Cast Data as integer (or integer array)
    CAUTION: WIP
    """

    def __init__(self, module: tp.Any) -> None:
        super().__init__(lower=0, upper=module)
        if not isinstance(module, (np.ndarray, np.float, np.int, float, int)):
            raise TypeError(f"Unsupported type {type(module)} for module")
        self._module = module

    def _layered_get_value(self) -> np.ndarray:
        return super()._layered_get_value() % self._module  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        current = super()._layered_get_value()
        super()._layered_set_value(current - (current % self._module) + value)
