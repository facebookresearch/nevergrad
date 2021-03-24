# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import warnings
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import utils
from ._layering import ValueProperty as ValueProperty
from ._layering import Layered as Layered
from ._layering import Level as Level


# pylint: disable=no-value-for-parameter,pointless-statement,import-outside-toplevel


P = tp.TypeVar("P", bound="Parameter")


# pylint: disable=too-many-public-methods
class Parameter(Layered):
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

    _LAYER_LEVEL = Level.ROOT
    value: ValueProperty[tp.Any, tp.Any] = ValueProperty()

    def __init__(self) -> None:
        # Main features
        super().__init__()
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
        self._constraint_checkers: tp.List[tp.Callable[[tp.Any], tp.Union[bool, float]]] = []
        self._name: tp.Optional[str] = None
        self._frozen = False
        self._meta: tp.Dict[tp.Hashable, tp.Any] = {}  # for anything algorithm related
        self.function = utils.FunctionInfo()

    @property
    def descriptors(self) -> utils.DeprecatedDescriptors:  # TODO remove
        return utils.DeprecatedDescriptors(self)

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
        self.set_standardized_data(self.random_state.normal(size=self.dimension))

    def sample(self: P) -> P:
        """Sample a new instance of the parameter.
        This usually means spawning a child and mutating it.
        This function should be used in optimizers when creating an initial population,
        and parameter.heritage["lineage"] is reset to parameter.uid instead of its parent's
        """
        # inner working can be overrided by _layer_sample()
        self.random_state  # make sure to populate it before copy
        child = self._layers[-1]._layered_sample()
        if not isinstance(child, Parameter) and not isinstance(child, type(self)):
            raise errors.NevergradRuntimeError("Unexpected sample return type")
        child._set_parenthood(None)
        return child  # type: ignore

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

    def set_standardized_data(self: P, data: tp.ArrayLike, *, reference: tp.Optional[P] = None) -> P:
        """Updates the value of the provided reference (or self) using the standardized data.

        Parameters
        ----------
        np.ndarray
            the representation of the value in the optimization space
        reference: Parameter
            the reference point for representing the data ("self", if not provided)

        Returns
        -------
        Parameter
            self (modified)

        Note
        ----
        To make the code more explicit, the "reference" is enforced
        as keyword-only parameters.
        """
        sent_reference = self if reference is None else reference
        assert isinstance(
            sent_reference, self.__class__
        ), f"Expected {type(self)} but got {type(sent_reference)} as reference"
        self._check_frozen()
        del self.value  # remove all cached information
        self._internal_set_standardized_data(np.array(data, copy=False), reference=sent_reference)
        return self

    def _internal_set_standardized_data(  # pylint: disable=unused-argument
        self: P, data: np.ndarray, reference: P
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
            return val.tobytes()
        else:
            raise errors.UnsupportedParameterOperationError(
                f"Value hash is not supported for object {self.name}"
            )

    def __repr__(self) -> str:
        strings = [self.name]
        if not callable(self.value):  # not a mutation
            strings.append(str(self.value))
        return ":".join(strings)

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
        self,
        func: tp.Union[tp.Callable[[tp.Any], bool], tp.Callable[[tp.Any], float]],
        as_layer: bool = False,
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
        - this function has an additional "as_layer" parameter which is experimental for now, and can have unexpected
          behavior
        """
        if getattr(func, "__name__", "not lambda") == "<lambda>":  # LambdaType does not work :(
            warnings.warn("Lambda as constraint is not advised because it may not be picklable.")
        if not as_layer:
            self._constraint_checkers.append(func)
        else:
            from nevergrad.ops.constraints import Constraint
            import nevergrad as ng

            compat_func = (
                func
                if not isinstance(self, ng.p.Instrumentation)
                else utils._ConstraintCompatibilityFunction(func)
            )
            self.add_layer(Constraint(compat_func))  # type: ignore

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
        # make sure to initialize the random state  before spawning children
        self.random_state  # pylint: disable=pointless-statement
        child = self.copy()
        child._set_parenthood(self)
        if new_value is not None:
            child.value = new_value
        return child

    def copy(self: P) -> P:
        """Creates a full copy of the parameter (with new unique uid).
        Use spawn_child instead to make sure to add the parenthood information.
        """
        child = super().copy()
        child.uid = uuid.uuid4().hex
        child._frozen = False
        child._subobjects = self._subobjects.new(child)
        child._meta = {}
        child.parents_uids = list(self.parents_uids)
        child.heritage = dict(self.heritage)
        child.loss = None
        child._losses = None
        child._constraint_checkers = list(self._constraint_checkers)
        # layers
        if self is not self._layers[0]:
            raise errors.NevergradRuntimeError("Something has gone horribly wrong with the layers")
        # subparameters
        attribute = self._subobjects.attribute
        container = getattr(child, attribute)
        if attribute != "__dict__":  # make a copy of the container if different from __dict__
            container = dict(container) if isinstance(container, dict) else list(container)
            setattr(child, attribute, container)
        for key, val in self._subobjects.items():
            container[key] = val.copy()
        del child.value  # clear cache
        return child

    def _set_parenthood(self, parent: tp.Optional["Parameter"]) -> None:
        """Sets the parenthood information to Parameter and subparameters."""
        if parent is None:
            self._generation = 0
            self.heritage = dict(lineage=self.uid)
            self.parents_uids = []
        else:
            self._generation = parent.generation + 1
            self.parents_uids = [parent.uid]
        self._subobjects.apply("_set_parenthood", parent)

    def freeze(self) -> None:
        """Prevents the parameter from changing value again (through value, mutate etc...)"""
        self._frozen = True
        self._subobjects.apply("freeze")

    def _check_frozen(self) -> None:
        if self._frozen and not isinstance(
            self, Constant
        ):  # nevermind constants (since they dont spawn children)
            raise RuntimeError(
                f"Cannot modify frozen Parameter {self.name}, please spawn a child and modify it instead"
                "(optimizers freeze the parametrization and all asked and told candidates to avoid border effects)"
            )
        self._subobjects.apply("_check_frozen")


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

    def _layered_get_value(self) -> tp.Any:
        return self._value

    def _layered_set_value(self, value: tp.Any) -> None:
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

    def _layered_sample(self: P) -> P:
        return self

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


class Operator(Layered):
    """Layer object that can be used as an operator on a Parameter"""

    _LAYER_LEVEL = Level.OPERATION

    def __call__(self, parameter: Parameter) -> Parameter:
        """Applies the operator on a Parameter to create a new Parameter"""
        new = parameter.copy()
        new.add_layer(self.copy())
        return new
