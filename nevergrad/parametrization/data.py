# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.(an
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import warnings
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors

from . import _layering
from . import core
from .container import Dict
from . import utils


# pylint: disable=no-value-for-parameter,import-outside-toplevel
# pylint: disable=cyclic-import


D = tp.TypeVar("D", bound="Data")
P = tp.TypeVar("P", bound=core.Parameter)


def _param_string(parameters: Dict) -> str:
    """Hacky helper for nice-visualizatioon"""
    substr = f"[{parameters._get_parameters_str()}]"
    if substr == "[]":
        substr = ""
    return substr


MutFn = tp.Callable[[tp.Sequence["Data"]], None]


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
    value: core.ValueProperty[MutFn, MutFn] = core.ValueProperty()

    def __init__(self, **kwargs: tp.Any) -> None:
        super().__init__()
        self.parameters = Dict(**kwargs)

    def _layered_get_value(self) -> MutFn:
        return self.apply

    def _layered_set_value(self, value: tp.Any) -> None:
        raise RuntimeError("Mutation cannot be set.")

    def _get_name(self) -> str:
        return super()._get_name() + _param_string(self.parameters)

    def apply(self, arrays: tp.Sequence["Data"]) -> None:
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
    lower: array, float or None
        minimum value
    upper: array, float or None
        maximum value
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    - More specific behaviors can be obtained throught the following methods:
      set_bounds, set_mutation, set_integer_casting
    - if both lower and upper bounds are provided, sigma will be adapted so that the range spans 6 sigma.
      Also, if init is not provided, it will be set to the middle value.
    """

    def __init__(
        self,
        *,
        init: tp.Optional[tp.ArrayLike] = None,
        shape: tp.Optional[tp.Tuple[int, ...]] = None,
        lower: tp.BoundValue = None,
        upper: tp.BoundValue = None,
        mutable_sigma: bool = False,
    ) -> None:
        super().__init__()
        sigma: tp.Any = np.array([1.0])
        # make sure either shape or
        if sum(x is None for x in [init, shape]) != 1:
            raise ValueError('Exactly one of "init" or "shape" must be provided')
        if init is not None:
            init = np.array(init, dtype=float, copy=False)
        else:
            assert isinstance(shape, (list, tuple)) and all(
                isinstance(n, int) for n in shape
            ), f"Incorrect shape: {shape}."
            init = np.zeros(shape, dtype=float)
            if lower is not None and upper is not None:
                init += (lower + upper) / 2.0
        self._value = init
        self.add_layer(_layering.ArrayCasting())
        # handle bounds
        num_bounds = sum(x is not None for x in (lower, upper))
        layer: tp.Any = None
        if num_bounds:
            from . import _datalayers

            layer = _datalayers.Bound(
                lower=lower, upper=upper, uniform_sampling=init is None and num_bounds == 2
            )
            if num_bounds == 2:
                sigma = (layer.bounds[1] - layer.bounds[0]) / 6
        # set parameters
        sigma = sigma[0] if sigma.size == 1 else sigma
        if mutable_sigma:
            siginit = sigma
            # for the choice of the base:
            # cf Evolution Strategies, Hans-Georg Beyer (2007)
            # http://www.scholarpedia.org/article/Evolution_strategies
            # we want:
            # sigma *= exp(gaussian / sqrt(dim))
            base = float(np.exp(1.0 / np.sqrt(2 * init.size)))
            sigma = base ** (Array if isinstance(sigma, np.ndarray) else Scalar)(
                init=siginit, mutable_sigma=False
            )
            sigma.value = siginit
        self.parameters = Dict(sigma=sigma, recombination="average", mutation="gaussian")
        self.parameters._ignore_in_repr = dict(sigma="1.0", recombination="average", mutation="gaussian")
        if layer is not None:
            layer(self, inplace=True)

    @property
    def bounds(self) -> tp.Tuple[tp.Optional[np.ndarray], tp.Optional[np.ndarray]]:
        """Estimate of the bounds (None if unbounded)

        Note
        ----
        This may be inaccurate (WIP)
        """
        from . import _datalayers

        bound_layers = _datalayers.BoundLayer.filter_from(self)
        if not bound_layers:
            return (None, None)
        bounds = bound_layers[-1].bounds
        forwardable = _datalayers.ForwardableOperation.filter_from(self)
        forwardable = [x for x in forwardable if x._layer_index > bound_layers[-1]._layer_index]
        for f in forwardable:
            bounds = tuple(None if b is None else f.forward(b) for b in bounds)  # type: ignore
        return bounds

    @property
    def dimension(self) -> int:
        return int(np.prod(self._value.shape))

    def _get_name(self) -> str:
        cls = self.__class__.__name__
        descriptors: tp.List[str] = []
        if self._value.shape != (1,):
            descriptors.append(str(self._value.shape).replace(" ", ""))
        descriptors += [
            layer.name
            for layer in self._layers[1:]
            if not isinstance(layer, (_layering.ArrayCasting, _layering._ScalarCasting))
        ]
        description = "" if not descriptors else "{{{}}}".format(",".join(descriptors))
        return f"{cls}{description}" + _param_string(self.parameters)

    @property
    def sigma(self) -> "Data":
        """Value for the standard deviation used to mutate the parameter"""
        return self.parameters["sigma"]  # type: ignore

    def _layered_sample(self: D) -> D:
        child = self.spawn_child()
        from . import helpers

        with helpers.deterministic_sampling(child):
            child.mutate()
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
        lower: array, float or None
            minimum value
        upper: array, float or None
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
        from . import _datalayers

        # if method == "constraint":
        #     method = "clipping"
        value = self.value
        if method == "constraint":
            layer = _datalayers.BoundLayer(lower=lower, upper=upper, uniform_sampling=full_range_sampling)
            checker = utils.BoundChecker(*layer.bounds)
            self.register_cheap_constraint(checker)
        else:
            layer = _datalayers.Bound(
                lower=lower, upper=upper, method=method, uniform_sampling=full_range_sampling
            )
        layer._LEGACY = True
        layer(self, inplace=True)
        _fix_legacy(self)
        try:
            self.value = value
        except ValueError as e:
            raise errors.NevergradValueError(
                "Current value is not within bounds, please update it first"
            ) from e
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
                new_state = func(size=self.dimension)
                self.set_standardized_data(new_state)
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
                self.sigma.value = sigma
        if exponent is not None:
            from . import _datalayers

            if exponent <= 0.0:
                raise ValueError("Only exponents strictly higher than 0.0 are allowed")
            value = self.value
            layer = _datalayers.Exponent(base=exponent)
            layer._LEGACY = True
            self.add_layer(layer)
            _fix_legacy(self)
            try:
                self.value = value
            except ValueError as e:
                raise errors.NevergradValueError(
                    "Cannot convert to logarithmic mode with current non-positive value, please update it firstp."
                ) from e
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
        return self.add_layer(_layering.Int())

    @property
    def integer(self) -> bool:
        return any(isinstance(x, _layering.Int) for x in self._layers)

    # pylint: disable=unused-argument
    def _internal_set_standardized_data(self: D, data: np.ndarray, reference: D) -> None:
        assert isinstance(data, np.ndarray)
        sigma = reference.sigma.value
        data_reduc = sigma * (data + reference._to_reduced_space()).reshape(reference._value.shape)
        self._value = data_reduc
        # make sure _value is updated by the layers getters if need be:
        self.value  # pylint: disable=pointless-statement

    def _internal_get_standardized_data(self: D, reference: D) -> np.ndarray:
        return reference._to_reduced_space(self._value - reference._value)

    def _to_reduced_space(self, value: tp.Optional[np.ndarray] = None) -> np.ndarray:
        """Converts array with appropriate shapes to reduced (uncentered) space
        by applying log scaling and sigma scaling
        """
        # TODO this is nearly useless now that the layer system has been added. Remove?
        if value is None:
            value = self._value
        reduced = value / self.sigma.value
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
            self.set_standardized_data(np.mean(all_arrays, axis=0))
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
        if self._value.shape != value.shape:
            raise ValueError(
                f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}"
            )
        self._value = value

    def _layered_get_value(self) -> np.ndarray:
        return self._value

    def _new_with_data_layer(self: D, name: str, *args: tp.Any, **kwargs: tp.Any) -> D:
        # pylint: disable=cyclic-import
        from . import _datalayers  # lazy to avoid cyclic imports

        new = self.copy()
        new.add_layer(getattr(_datalayers, name)(*args, **kwargs))
        return new

    def __mod__(self: D, module: tp.Any) -> D:
        return self._new_with_data_layer("Modulo", module)

    def __rpow__(self: D, base: float) -> D:
        return self._new_with_data_layer("Exponent", base)

    def __add__(self: D, offset: tp.Any) -> D:
        return self._new_with_data_layer("Add", offset)

    def __sub__(self: D, offset: tp.Any) -> D:
        return self.__add__(-offset)

    def __radd__(self: D, offset: tp.Any) -> D:
        return self.__add__(offset)

    def __mul__(self: D, value: tp.Any) -> D:
        return self._new_with_data_layer("Multiply", value)

    def __rmul__(self: D, value: tp.Any) -> D:
        return self.__mul__(value)

    def __truediv__(self: D, value: tp.Any) -> D:
        return self.__mul__(1.0 / value)

    def __rtruediv__(self: D, value: tp.Any) -> D:
        return value * (self ** -1)  # type: ignore

    def __pow__(self: D, power: float) -> D:
        return self._new_with_data_layer("Power", power)

    def __neg__(self: D) -> D:
        return self.__mul__(-1.0)


def _fix_legacy(parameter: Data) -> None:
    """Ugly hack for keeping the legacy behaviors with considers bounds always after the exponent
    and can still sample "before" the exponent (log-uniform).
    """
    from . import _datalayers

    legacy = [x for x in _datalayers.Operation.filter_from(parameter) if x._LEGACY]
    if len(legacy) < 2:
        return
    if len(legacy) > 2:
        raise errors.NevergradRuntimeError("More than 2 legacy layers, this should not happen, open an issue")
    # warnings.warn(
    #     "Settings bounds and exponent through the Array/Scalar API will change behavior "
    #     " (this is an early warning, more on this asap)",
    #     errors.NevergradBehaviorChangesWarning,
    # )  # TODO activate when ready
    value = parameter.value
    layers_inds = tuple(leg._layer_index for leg in legacy)
    if abs(layers_inds[0] - layers_inds[1]) > 1:
        raise errors.NevergradRuntimeError("Non-legacy layers between 2 legacy layers")
    parameter._layers = [x for x in parameter._layers if x._layer_index not in layers_inds]
    # fix parameter layers
    for k, sub in enumerate(parameter._layers):
        sub._layer_index = k
        sub._layers = parameter._layers
    parameter.value = value
    bound_ind = int(isinstance(legacy[0], _datalayers.Exponent))
    bound: _datalayers.BoundLayer = legacy[bound_ind]  # type: ignore
    exp: _datalayers.Exponent = legacy[(bound_ind + 1) % 2]  # type: ignore
    bound.bounds = tuple(None if b is None else exp.backward(b) for b in bound.bounds)  # type: ignore
    if isinstance(bound, _datalayers.Bound):
        bound = _datalayers.Bound(
            lower=bound.bounds[0],
            upper=bound.bounds[1],
            method=bound._method,
            uniform_sampling=bound.uniform_sampling,
        )
    for l in (bound, exp):
        l._layer_index = 0
        l._layers = [l]
        parameter.add_layer(l)
    return


class Array(Data):

    value: core.ValueProperty[tp.ArrayLike, np.ndarray] = core.ValueProperty()


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

    value: core.ValueProperty[float, float] = core.ValueProperty()

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
        from . import _datalayers

        exp_layer = _datalayers.Exponent(exponent)
        raw_bounds = tuple(None if x is None else np.array([x], dtype=float) for x in (lower, upper))
        bounds = tuple(None if x is None else exp_layer.backward(x) for x in raw_bounds)
        init = exp_layer.backward(np.array([init]))
        super().__init__(init=init[0], mutable_sigma=mutable_sigma)  # type: ignore
        # TODO remove the next line when all compatibility is done
        if any(x is not None for x in bounds):
            bound_layer = _datalayers.Bound(*bounds, uniform_sampling=bounded and no_init)  # type: ignore
            bound_layer(self, inplace=True)
        self.add_layer(exp_layer)
