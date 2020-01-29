# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import typing as tp
import numpy as np
from nevergrad.common.typetools import ArrayLike
from . import core
from . import utils
from . import transforms as trans
# pylint: disable=no-value-for-parameter


BoundValue = tp.Optional[tp.Union[float, int, np.int, np.float, np.ndarray]]
A = tp.TypeVar("A", bound="Array")


class BoundChecker:
    """Simple object for checking whether an array lies
    between provided bounds.

    Parameter
    ---------
    a_min: float or None
        minimum value
    a_max: float or None
        maximum value

    Note
    -----
    Not all bounds are necessary (data can be partially bounded, or not at all actually)
    """

    def __init__(self, a_min: BoundValue = None, a_max: BoundValue = None) -> None:
        self.bounds = (a_min, a_max)

    def __call__(self, value: np.ndarray) -> bool:
        """Checks whether the array lies within the bounds

        Parameter
        ---------
        value: np.ndarray
            array to check

        Returns
        -------
        bool
            True iff the array lies within the bounds
        """
        for k, bound in enumerate(self.bounds):
            if bound is not None:
                if np.any((value > bound) if k else (value < bound)):
                    return False
        return True


# pylint: disable=too-many-arguments, too-many-instance-attributes
class Array(core.Parameter):
    """Array variable of a given shape.

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
            init: tp.Optional[ArrayLike] = None,
            shape: tp.Optional[tp.Tuple[int, ...]] = None,
            mutable_sigma: bool = False
    ) -> None:
        sigma = Log(init=1.0, exponent=1.2, mutable_sigma=False) if mutable_sigma else 1.0
        super().__init__(sigma=sigma, recombination="average")
        err_msg = 'Exactly one of "init" or "shape" must be provided'
        if init is not None:
            if shape is not None:
                raise ValueError(err_msg)
            self._value = np.array(init, copy=False)
        elif shape is not None:
            assert isinstance(shape, tuple) and all(isinstance(n, int) for n in shape)
            self._value = np.zeros(shape)
        else:
            raise ValueError(err_msg)
        self.integer = False
        self.exponent: tp.Optional[float] = None
        self.bounds: tp.Tuple[tp.Optional[np.ndarray], tp.Optional[np.ndarray]] = (None, None)
        self.bound_transform: tp.Optional[trans.BoundTransform] = None
        self.full_range_sampling = False
        self._ref_data: tp.Optional[np.ndarray] = None

    def _compute_descriptors(self) -> utils.Descriptors:
        return utils.Descriptors(continuous=not self.integer)

    def _get_name(self) -> str:
        cls = self.__class__.__name__
        descriptors: tp.List[str] = (["int"] if self.integer else
                                     ([str(self.value.shape).replace(" ", "")] if self.value.shape != () else []))
        descriptors += [f"exp={self.exponent}"] if self.exponent is not None else []
        descriptors += [f"{self.bound_transform}"] if self.bound_transform is not None else []
        descriptors += ["constr"] if self._constraint_checkers else []
        description = ""
        if descriptors:
            description = "{{{}}}".format(",".join(descriptors))
        return f"{cls}{description}"

    @property
    def sigma(self) -> tp.Union["Array", "Scalar"]:
        """Value for the standard deviation used to mutate the parameter
        """
        return self.parameters["sigma"]  # type: ignore

    @property
    def value(self) -> np.ndarray:
        if self.integer:
            return np.round(self._value)  # type: ignore
        return self._value

    @value.setter
    def value(self, value: ArrayLike) -> None:
        self._check_frozen()
        self._ref_data = None
        if not isinstance(value, (np.ndarray, tuple, list)):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray/tuple/list")
        value = np.asarray(value)
        assert isinstance(value, np.ndarray)
        if self._value.shape != value.shape:
            raise ValueError(f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}")
        if not BoundChecker(*self.bounds)(self.value):
            raise ValueError("New value does not comply with bounds")
        if self.exponent is not None and np.min(value.ravel()) <= 0:
            raise ValueError("Logirithmic values cannot be negative")
        self._value = value

    def sample(self: A) -> A:
        if not self.full_range_sampling:
            return super().sample()
        child = self.spawn_child()
        func = (lambda x: x) if self.exponent is None else self._to_reduced_space  # noqa
        std_bounds = tuple(func(b * np.ones(self.value.shape)) for b in self.bounds)
        diff = std_bounds[1] - std_bounds[0]
        new_data = std_bounds[0] + np.random.uniform(0, 1, size=diff.shape) * diff
        if self.exponent is None:
            new_data = self._to_reduced_space(new_data)
        child.set_standardized_data(new_data - self._get_ref_data(), deterministic=False)
        child.heritage["lineage"] = child.uid
        return child

    def set_bounds(self: A, a_min: BoundValue = None, a_max: BoundValue = None,
                   method: str = "clipping", full_range_sampling: bool = False) -> A:
        """Bounds all real values into [a_min, a_max] using a provided method

        Parameters
        ----------
        a_min: float or None
            minimum value
        a_max: float or None
            maximum value
        method: str
            One of the following choices:
            - "clipping": clips the values inside the bounds. This is efficient but leads
              to over-sampling on the bounds.
            - "constraint": adds a constraint (see register_cheap_constraint) which leads to rejecting mutations
              reaching beyond the bounds. This avoids oversampling the boundaries, but can be inefficient in large
              dimension.
            - "arctan": maps the space [a_min, a_max] to to all [-inf, inf] using arctan transform. This is efficient
              but it completely reshapes the space (a mutation in the center of the space will be larger than a mutation
              close to the bounds), and reaching the bounds is equivalent to reaching the infinity.
            - "tanh": same as "arctan", but with a "tanh" transform. "tanh" saturating much faster than "arctan", it can lead
              to unexpected behaviors.
        full_range_sampling: bool
            this changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            to creating a child with a value sampled uniformly (or log-uniformly) within the while range of the bounds. The
            "sample" method is used by some algorithms to create an initial population.

        Notes
        -----
        - "tanh" reaches the boundaries really quickly, while "arctan" is much softer
        - only "clipping" accepts partial bounds (None values)
        """  # TODO improve description of methods
        bounds = tuple(a if isinstance(a, np.ndarray) or a is None else np.array([a], dtype=float) for a in (a_min, a_max))
        both_bounds = all(b is not None for b in bounds)
        # preliminary checks
        if self.bound_transform is not None:
            raise RuntimeError("A bounding method has already been set")
        if full_range_sampling and not both_bounds:
            raise ValueError("Cannot use full range sampling if both bounds are not set")
        checker = BoundChecker(*bounds)
        if not checker(self.value):
            raise ValueError("Current value is not within bounds, please update it first")
        if not (a_min is None or a_max is None):
            if (bounds[0] >= bounds[1]).any():  # type: ignore
                raise ValueError(f"Lower bounds {a_min} should be strictly smaller than upper bounds {a_max}")
        # update instance
        transforms = dict(clipping=trans.Clipping, arctan=trans.ArctanBound, tanh=trans.TanhBound)
        if method in transforms:
            if self.exponent is not None and method != "clipping":
                raise ValueError(f'Cannot use method "{method}" in logarithmic mode')
            self.bound_transform = transforms[method](*bounds)
        elif method == "constraint":
            self.register_cheap_constraint(checker)
        else:
            raise ValueError(f"Unknown method {method}")
        self.bounds = bounds  # type: ignore
        self.full_range_sampling = full_range_sampling
        # warn if sigma is too large for range
        if both_bounds and method != "tanh":  # tanh goes to infinity anyway
            std_bounds = tuple(self._to_reduced_space(b) for b in self.bounds)  # type: ignore
            min_dist = np.min(np.abs(std_bounds[0] - std_bounds[1]).ravel())
            if min_dist < 3.0:
                warnings.warn(f"Bounds are {min_dist} sigma away from each other at the closest, "
                              "you should aim for at least 3 for better quality.")
        return self

    def set_recombination(self: A, recombination: tp.Union[str, core.Parameter, utils.Crossover]) -> A:
        assert self._parameters is not None
        self._parameters._content["recombination"] = (recombination if isinstance(recombination, core.Parameter)
                                                      else core.Constant(recombination))
        return self

    def set_mutation(self: A, sigma: tp.Optional[tp.Union[float, "Array"]] = None, exponent: tp.Optional[float] = None) -> A:
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

        Returns
        -------
        self
        """
        if sigma is not None:
            # just replace if an actual Parameter is provided as sigma, else update value (parametrized or not)
            if isinstance(sigma, core.Parameter) or isinstance(self.parameters._content["sigma"], core.Constant):
                self.parameters._content["sigma"] = core.as_parameter(sigma)
            else:
                self.sigma.value = sigma  # type: ignore
        if exponent is not None:
            if self.bound_transform is not None and not isinstance(self.bound_transform, trans.Clipping):
                raise RuntimeError(f"Cannot set logarithmic transform with bounding transform {self.bound_transform}, "
                                   "only clipping and constraint bounding methods can accept itp.")
            if exponent <= 1.0:
                raise ValueError("Only exponents strictly higher than 1.0 are allowed")
            if np.min(self._value.ravel()) <= 0:
                raise RuntimeError("Cannot convert to logarithmic mode with current non-positive value, please update it firstp.")
            self.exponent = exponent
        return self

    def set_integer_casting(self: A) -> A:
        """Output will be cast to integer(s) through deterministic rounding.

        Returns
        -------
        self
        """
        self.integer = True
        return self

    # pylint: disable=unused-argument
    def _internal_set_standardized_data(self: A, data: np.ndarray, reference: A, deterministic: bool = False) -> None:
        assert isinstance(data, np.ndarray)
        sigma = reference.sigma.value
        data_reduc = sigma * (data + reference._get_ref_data()).reshape(reference._value.shape)
        self._value = data_reduc if reference.exponent is None else reference.exponent**data_reduc
        self._ref_data = None
        if reference.bound_transform is not None:
            self._value = reference.bound_transform.forward(self._value)

    def _internal_spawn_child(self) -> "Array":
        child = self.__class__(init=self.value)
        child.parameters._content = {k: v.spawn_child() if isinstance(v, core.Parameter) else v
                                     for k, v in self.parameters._content.items()}
        for name in ["integer", "exponent", "bounds", "bound_transform", "full_range_sampling"]:
            setattr(child, name, getattr(self, name))
        return child

    def _internal_get_standardized_data(self: A, reference: A) -> np.ndarray:
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

    def recombine(self: A, *others: A) -> None:
        if not others:
            return
        recomb = self.parameters["recombination"].value
        all_params = [self] + list(others)
        if isinstance(recomb, str) and recomb == "average":
            all_arrays = [p.get_standardized_data(reference=self) for p in all_params]
            self.set_standardized_data(np.mean(all_arrays, axis=0), deterministic=False)
        elif isinstance(recomb, utils.Crossover):
            crossover = recomb.apply([p.value for p in all_params], self.random_state)
            self.value = crossover
        else:
            raise ValueError(f'Unknown recombination "{recomb}"')


class Scalar(Array):
    """Parameter representing a scalar

    Parameters
    ----------
    init: float
        initial value of the scalar
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    More specific behaviors can be obtained throught the following methods:
    set_bounds, set_mutation
    """

    def __init__(self, init: float = 0.0, mutable_sigma: bool = True) -> None:
        super().__init__(init=np.array([init]), mutable_sigma=mutable_sigma)

    @property  # type: ignore
    def value(self) -> float:  # type: ignore
        return self._value[0] if not self.integer else int(self._value[0])  # type: ignore

    @value.setter
    def value(self, value: float) -> None:
        self._check_frozen()
        if not isinstance(value, (float, int, np.float, np.int)):
            raise TypeError(f"Received a {type(value)} in place of a scalar (float, int)")
        self._value = np.array([value], dtype=float)


class Log(Scalar):
    """Parameter representing a log distributed value between 0 and infinity

    Parameters
    ----------
    init: float or None
        initial value of the variable. If not provided, it is set to the middle of a_min and a_max in log space
    exponent: float or None
        exponent for the log mutation: an exponent of 2.0 will lead to mutations by factors between around 0.5 and 2.0
        By default, it is set to either 2.0, or if the parameter is completely bounded to a factor so that bounds are
        at 5 sigma in the transformed space.
    a_min: float or None
        minimum value if any (> 0)
    a_max: float or None
        maximum value if any
    mutable_sigma: bool
        whether the mutation standard deviation must mutate as well (for mutation based algorithms)

    Note
    ----
    More specific behaviors can be obtained throught the following methods:
    set_bounds, set_mutation
    """

    def __init__(
        self,
        *,
        init: tp.Optional[float] = None,
        exponent: tp.Optional[float] = None,
        a_min: tp.Optional[float] = None,
        a_max: tp.Optional[float] = None,
        mutable_sigma: bool = False,
    ) -> None:
        if all(a is not None for a in (a_min, a_max)):
            if init is None:
                init = float(np.sqrt(a_min * a_max))  # type: ignore
            if exponent is None:
                exponent = float(np.exp((np.log(a_max) - np.log(a_min)) / 10.0))
        if init is None:
            raise ValueError("You must define either a init value or both a_min and a_max bounds")
        if exponent is None:
            exponent = 2.0
        super().__init__(init=init, mutable_sigma=mutable_sigma)
        self.set_mutation(sigma=1.0, exponent=exponent)
        if any(a is not None for a in (a_min, a_max)):
            self.set_bounds(a_min, a_max, method="clipping")
