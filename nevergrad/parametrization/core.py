# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import copy
import warnings
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import utils

# pylint: disable=no-value-for-parameter


P = tp.TypeVar("P", bound="Parameter")
X = tp.TypeVar("X")


class ValueProperty(tp.Generic[X]):
    """Typed property (descriptor) object so that the value attribute of
    Parameter objects fetches _get_value and _set_value methods
    """

    # This uses the descriptor protocol, like a property:
    # See https://docs.python.org/3/howto/descriptor.html
    #
    # Basically parameter.value calls parameter.value.__get__
    # and then parameter._get_value
    def __init__(self) -> None:
        self.__doc__ = """Value of the Parameter, which should be sent to the function
        to optimize.

        Example
        -------
        >>> ng.p.Array(shape=(2,)).value
        array([0., 0.])
        """

    def __get__(self, obj: "Parameter", objtype: tp.Optional[tp.Type[object]] = None) -> X:
        return obj._get_value()  # type: ignore

    def __set__(self, obj: "Parameter", value: X) -> None:
        obj._set_value(value)


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class Parameter:
    """Class providing the core functionality of a parameter, aka
    value, internal/model parameters, mutation, recombination
    and additional features such as shared random state,
    constraint check, hashes, generation and naming.
    The value field should sent to the function to optimize.

    Example
    -------
    >>> ng.p.Array(shape=(2,)).value
    array([0., 0.])
    """

    # By default, all Parameter attributes of this Parameter are considered as
    # sub-parameters.
    # Spawning a child creates a shallow copy.

    value: ValueProperty[tp.Any] = ValueProperty()

    def __init__(self) -> None:
        # Main features
        self.uid = uuid.uuid4().hex
        self._subobjects = utils.Subobjects(
            self, base=Parameter, attribute="__dict__"
        )  # registers and apply functions too all (sub-)Parameter attributes
        self.parents_uids: tp.List[str] = []
        self.heritage: tp.Dict[tp.Hashable, tp.Any] = {"lineage": self.uid}  # passed through to children
        self.loss: tp.Optional[float] = None  # associated loss
        self._losses: tp.Optional[np.ndarray] = None  # associated losses (multiobjective) as an array
        self._dimension: tp.Optional[int] = None
        # Additional convenient features
        self._random_state: tp.Optional[np.random.RandomState] = None  # lazy initialization
        self._generation = 0
        # self._constraint_checkers: tp.List[tp.Union[tp.Callable[[tp.Any], bool], tp.Callable[[tp.Any], float]]] = []
        self._constraint_checkers: tp.List[tp.Callable[[tp.Any], tp.Union[bool, float]]] = []
        self._name: tp.Optional[str] = None
        self._frozen = False
        self._descriptors: tp.Optional[utils.Descriptors] = None
        self._meta: tp.Dict[tp.Hashable, tp.Any] = {}  # for anything algorithm related

    @property
    def losses(self) -> np.ndarray:
        """Possibly multiobjective losses which were told
        to the optimizer along this parameter.
        In case of mono-objective loss, losses is the array containing this loss as sole element

        Note
        ----
        This API is highly experimental
        """
        if self._losses is not None:
            return self._losses
        if self.loss is not None:
            return np.array([self.loss], dtype=float)
        raise RuntimeError("No loss was provided")

    def _get_value(self) -> tp.Any:
        raise NotImplementedError

    def _set_value(self, value: tp.Any) -> tp.Any:
        raise NotImplementedError

    @property
    def args(self) -> tp.Tuple[tp.Any, ...]:
        """Value of the positional arguments.
        Used to input value in a function as `func(*param.args, **param.kwargs)`
        Use `parameter.Instrumentation` to set `args` and `kwargs` with full freedom.
        """
        return (self.value,)

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        """Value of the keyword arguments.
        Used to input value in a function as `func(*param.args, **param.kwargs)`
        Use `parameter.Instrumentation` to set `args` and `kwargs` with full freedom.
        """
        return {}

    @property
    def dimension(self) -> int:
        """Dimension of the standardized space for this parameter
        i.e size of the vector returned by get_standardized_data(reference=...)
        """
        if self._dimension is None:
            try:
                self._dimension = self.get_standardized_data(reference=self).size
            except errors.UnsupportedParameterOperationError:
                self._dimension = 0
        return self._dimension

    def mutate(self) -> None:
        """Mutate parameters of the instance, and then its value"""
        self._check_frozen()
        self._subobjects.apply("mutate")
        self.set_standardized_data(self.random_state.normal(size=self.dimension), deterministic=False)

    def sample(self: P) -> P:
        """Sample a new instance of the parameter.
        This usually means spawning a child and mutating it.
        This function should be used in optimizers when creating an initial population,
        and parameter.heritage["lineage"] is reset to parameter.uid instead of its parent's
        """
        child = self._inner_copy(mode="sample")
        child.mutate()
        return child

    def recombine(self: P, *others: P) -> None:
        """Update value and parameters of this instance by combining it with
        other instances.

        Parameters
        ----------
        *others: Parameter
            other instances of the same type than this instance.
        """
        if not others:
            return
        self.random_state  # pylint: disable=pointless-statement
        assert all(isinstance(o, self.__class__) for o in others)
        self._subobjects.apply("recombine", *others)

    def get_standardized_data(self: P, *, reference: P) -> np.ndarray:
        """Get the standardized data representing the value of the instance as an array in the optimization space.
        In this standardized space, a mutation is typically centered and reduced (sigma=1) Gaussian noise.
        The data only represent the value of this instance, not the parameters (eg.: mutable sigma), hence it does not
        fully represent the state of the instance. Also, in stochastic cases, the value can be non-deterministically
        deduced from the data (eg.: categorical variable, for which data includes sampling weights for each value)

        Parameters
        ----------
        reference: Parameter
            the reference instance for representation in the standardized data space. This keyword parameter is
            mandatory to make the code clearer.
            If you use "self", this method will always return a zero vector.

        Returns
        -------
        np.ndarray
            the representation of the value in the optimization space

        Note
        ----
        - Operations between different standardized data should only be performed if each array was produced
          by the same reference in the exact same state (no mutation)
        - to make the code more explicit, the "reference" parameter is enforced as a keyword-only parameter.
        """
        assert reference is None or isinstance(
            reference, self.__class__
        ), f"Expected {type(self)} but got {type(reference)} as reference"
        return self._internal_get_standardized_data(self if reference is None else reference)

    def _internal_get_standardized_data(self: P, reference: P) -> np.ndarray:
        raise errors.UnsupportedParameterOperationError(
            f"Export to standardized data space is not implemented for {self.name}"
        )

    def set_standardized_data(
        self: P, data: tp.ArrayLike, *, reference: tp.Optional[P] = None, deterministic: bool = False
    ) -> P:
        """Updates the value of the provided reference (or self) using the standardized data.

        Parameters
        ----------
        np.ndarray
            the representation of the value in the optimization space
        reference: Parameter
            the reference point for representing the data ("self", if not provided)
        deterministic: bool
            whether the value should be deterministically drawn (max probability) in the case of stochastic parameters

        Returns
        -------
        Parameter
            self (modified)

        Note
        ----
        To make the code more explicit, the "reference" and "deterministic" parameters are enforced
        as keyword-only parameters.
        """
        assert isinstance(deterministic, bool)
        sent_reference = self if reference is None else reference
        assert isinstance(
            sent_reference, self.__class__
        ), f"Expected {type(self)} but got {type(sent_reference)} as reference"
        self._check_frozen()
        self._internal_set_standardized_data(
            np.array(data, copy=False), reference=sent_reference, deterministic=deterministic
        )
        return self

    def _internal_set_standardized_data(  # pylint: disable=unused-argument
        self: P, data: np.ndarray, reference: P, deterministic: bool = False
    ) -> None:
        if data.size:
            raise errors.UnsupportedParameterOperationError(
                f"Import from standardized data space is not implemented for {self.name}"
            )

    # PART 2 - Additional features

    @property
    def generation(self) -> int:
        """Generation of the parameter (children are current generation + 1)"""
        return self._generation

    def get_value_hash(self) -> tp.Hashable:
        """Hashable object representing the current value of the instance"""
        val = self.value
        if isinstance(val, (str, bytes, float, int)):
            return val
        elif isinstance(val, np.ndarray):
            return val.tobytes()  # type: ignore
        else:
            raise errors.UnsupportedParameterOperationError(
                f"Value hash is not supported for object {self.name}"
            )

    def _get_name(self) -> str:
        """Internal implementation of parameter name. This should be value independant, and should not account
        for internal/model parameters.
        """
        return self.__class__.__name__

    @property
    def name(self) -> str:
        """Name of the parameter
        This is used to keep track of how this Parameter is configured (included through internal/model parameters),
        mostly for reproducibility A default version is always provided, but can be overriden directly
        through the attribute, or through the set_name method (which allows chaining).
        """
        if self._name is not None:
            return self._name
        return self._get_name()

    @name.setter
    def name(self, name: str) -> None:
        self.set_name(name)  # with_name allows chaining

    def __repr__(self) -> str:
        strings = [self.name]
        if not callable(self.value):  # not a mutation
            strings.append(str(self.value))
        return ":".join(strings)

    def set_name(self: P, name: str) -> P:
        """Sets a name and return the current instrumentation (for chaining)

        Parameters
        ----------
        name: str
            new name to use to represent the Parameter
        """
        self._name = name
        return self

    # %% Constraint management

    def satisfies_constraints(self) -> bool:
        """Whether the instance satisfies the constraints added through
        the `register_cheap_constraint` method

        Returns
        -------
        bool
            True iff the constraint is satisfied
        """
        inside = self._subobjects.apply("satisfies_constraints")
        if not all(inside.values()):
            return False
        if not self._constraint_checkers:
            return True
        val = self.value
        return all(utils.float_penalty(func(val)) <= 0 for func in self._constraint_checkers)

    def register_cheap_constraint(
        self, func: tp.Union[tp.Callable[[tp.Any], bool], tp.Callable[[tp.Any], float]]
    ) -> None:
        """Registers a new constraint on the parameter values.

        Parameters
        ----------
        func: Callable
            function which, given the value of the instance, returns whether it satisfies the constraints (if output = bool),
            or a float which is >= 0 if the constraint is satisfied.

        Note
        ----
        - this is only for checking after mutation/recombination/etc if the value still satisfy the constraints.
          The constraint is not used in those processes.
        - constraints should be fast to compute.
        """
        if getattr(func, "__name__", "not lambda") == "<lambda>":  # LambdaType does not work :(
            warnings.warn("Lambda as constraint is not advised because it may not be picklable.")
        self._constraint_checkers.append(func)

    # %% random state

    @property
    def random_state(self) -> np.random.RandomState:
        """Random state the instrumentation and the optimizers pull from.
        It can be seeded/replaced.
        """
        if self._random_state is None:
            # use the setter, to make sure the random state is propagated to the variables
            seed = np.random.randint(2 ** 32, dtype=np.uint32)  # better way?
            self._set_random_state(np.random.RandomState(seed))
        assert self._random_state is not None
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        self._set_random_state(random_state)

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        self._random_state = random_state
        self._subobjects.apply("_set_random_state", random_state)

    def _inner_copy(self: P, mode: str) -> P:
        # make sure to initialize the random state  before spawning children
        if mode != "copy":
            self.random_state  # pylint: disable=pointless-statement
        child = copy.copy(self)
        child.uid = uuid.uuid4().hex
        child._frozen = False
        child._subobjects = self._subobjects.new(child)
        child._meta = {}
        child.parents_uids = list(self.parents_uids)
        child.heritage = dict(self.heritage)
        child.loss = None
        child._losses = None
        child._constraint_checkers = list(self._constraint_checkers)
        attribute = self._subobjects.attribute
        container = getattr(child, attribute)
        if attribute != "__dict__":  # make a copy of the container if different from __dict__
            container = dict(container) if isinstance(container, dict) else list(container)
            setattr(child, attribute, container)
        for key, val in self._subobjects.items():
            container[key] = val._inner_copy(mode=mode)
        if mode == "spawn_child":
            child._generation += 1
            child.parents_uids = [self.uid]
        elif mode == "sample":
            child._generation = 0
            child.heritage = dict(lineage=child.uid)
            child.parents_uids = []
        elif mode != "copy":
            raise NotImplementedError(f"No copy mode {mode!r}")
        return child

    def spawn_child(self: P, new_value: tp.Optional[tp.Any] = None) -> P:
        """Creates a new instance which shares the same random generator than its parent,
        is sampled from the same data, and mutates independently from the parentp.
        If a new value is provided, it will be set to the new instance

        Parameters
        ----------
        new_value: anything (optional)
            if provided, it will update the new instance value (cannot be used at the same time as new_data).

        Returns
        -------
        Parameter
            a new instance of the same class, with same content/internal-model parameters/...
            Optionally, a new value will be set after creation
        """
        child = self._inner_copy(mode="spawn_child")
        if new_value is not None:
            child.value = new_value
        return child

    def freeze(self) -> None:
        """Prevents the parameter from changing value again (through value, mutate etc...)"""
        self._frozen = True
        self._subobjects.apply("freeze")

    def _check_frozen(self) -> None:
        if self._frozen and not isinstance(
            self, Constant
        ):  # nevermind constants (since they dont spawn children)
            raise RuntimeError(
                f"Cannot modify frozen Parameter {self}, please spawn a child and modify it instead"
                "(optimizers freeze the parametrization and all asked and told candidates to avoid border effects)"
            )
        self._subobjects.apply("_check_frozen")

    def copy(self: P) -> P:  # TODO test (see former instrumentation_copy test)
        """Create a child, but remove the random state
        This is used to run multiple experiments
        """
        return self._inner_copy(mode="copy")

    def _compute_descriptors(self) -> utils.Descriptors:
        return utils.Descriptors()

    @property
    def descriptors(self) -> utils.Descriptors:
        if self._descriptors is None:
            self._compute_descriptors()
            self._descriptors = self._compute_descriptors()
        return self._descriptors


# Basic types and helpers #


class Constant(Parameter):
    """Parameter-like object for simplifying management of constant parameters:
    mutation/recombination do nothing, value cannot be changed, standardize data is an empty array,
    child is the same instance.

    Parameter
    ---------
    value: Any
        the value that this parameter will always provide
    """

    def __init__(self, value: tp.Any) -> None:
        super().__init__()
        if isinstance(value, Parameter) and not isinstance(self, MultiobjectiveReference):
            raise TypeError("Only non-parameters can be wrapped in a Constant")
        self._value = value

    def _get_name(self) -> str:
        return str(self._value)

    def get_value_hash(self) -> tp.Hashable:
        try:
            return super().get_value_hash()
        except errors.UnsupportedParameterOperationError:
            return "#non-hashable-constant#"

    def _get_value(self) -> tp.Any:
        return self._value

    def _set_value(self, value: tp.Any) -> None:
        different = False
        if isinstance(value, np.ndarray):
            if not np.equal(value, self._value).all():
                different = True
        elif not (value == self._value or value is self._value):
            different = True
        if different:
            raise ValueError(
                f'Constant value can only be updated to the same value (in this case "{self._value}")'
            )

    def get_standardized_data(  # pylint: disable=unused-argument
        self: P, *, reference: tp.Optional[P] = None
    ) -> np.ndarray:
        return np.array([])

    def spawn_child(self: P, new_value: tp.Optional[tp.Any] = None) -> P:
        if new_value is not None:
            self.value = new_value  # check that it is equal
        return self  # no need to create another instance for a constant

    def recombine(self: P, *others: P) -> None:
        pass

    def mutate(self) -> None:
        pass


def as_parameter(param: tp.Any) -> Parameter:
    """Returns a Parameter from anything:
    either the input if it is already a parameter, or a Constant if not
    This is convenient for iterating over Parameter and other objects alike
    """
    if isinstance(param, Parameter):
        return param
    else:
        return Constant(param)


class MultiobjectiveReference(Constant):
    def __init__(self, parameter: tp.Optional[Parameter] = None) -> None:
        if parameter is not None and not isinstance(parameter, Parameter):
            raise TypeError(
                "MultiobjectiveReference should either take no argument or a parameter which will "
                f"be used by the optimizer.\n(received {parameter} of type {type(parameter)})"
            )
        super().__init__(parameter)
