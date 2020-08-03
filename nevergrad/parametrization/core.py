# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import warnings
import operator
import functools
from collections import OrderedDict
import numpy as np
import nevergrad.common.typing as tp
from . import utils
# pylint: disable=no-value-for-parameter


P = tp.TypeVar("P", bound="Parameter")
D = tp.TypeVar("D", bound="Dict")


# pylint: disable=too-many-instance-attributes,too-many-public-methods
class Parameter:
    """Abstract class providing the core functionality of a parameter, aka
    value, internal/model parameters, mutation, recombination
    and additional features such as shared random state,
    constraint check, hashes, generation and naming.
    """

    def __init__(self, **parameters: tp.Any) -> None:
        # Main features
        self.uid = uuid.uuid4().hex
        self.parents_uids: tp.List[str] = []
        self.heritage: tp.Dict[str, tp.Any] = {"lineage": self.uid}  # passed through to children
        self.loss: tp.Optional[float] = None  # associated loss
        self._parameters = None if not parameters else Dict(**parameters)  # internal/model parameters
        self._dimension: tp.Optional[int] = None
        # Additional convenient features
        self._random_state: tp.Optional[np.random.RandomState] = None  # lazy initialization
        self._generation = 0
        self._constraint_checkers: tp.List[tp.Callable[[tp.Any], bool]] = []
        self._name: tp.Optional[str] = None
        self._frozen = False
        self._descriptors: tp.Optional[utils.Descriptors] = None
        self._meta: tp.Dict[str, tp.Any] = {}  # for anything algorithm related

    @property
    def value(self) -> tp.Any:
        raise NotImplementedError

    @value.setter
    def value(self, value: tp.Any) -> tp.Any:
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
    def parameters(self) -> "Dict":
        """Internal/model parameters for this parameter
        """
        if self._parameters is None:  # delayed instantiation to avoid infinte loop
            assert self.__class__ != Dict, "parameters of Parameters dict should never be called"
            self._parameters = Dict()
        assert self._parameters is not None
        return self._parameters

    @property
    def dimension(self) -> int:
        """Dimension of the standardized space for this parameter
        i.e size of the vector returned by get_standardized_data(reference=...)
        """
        if self._dimension is None:
            try:
                self._dimension = self.get_standardized_data(reference=self).size
            except utils.NotSupportedError:
                self._dimension = 0
        return self._dimension

    def mutate(self) -> None:
        """Mutate parameters of the instance, and then its value
        """
        self._check_frozen()
        self.parameters.mutate()
        self.set_standardized_data(self.random_state.normal(size=self.dimension), deterministic=False)

    def sample(self: P) -> P:
        """Sample a new instance of the parameter.
        This usually means spawning a child and mutating it.
        This function should be used in optimizers when creating an initial population,
        and parameter.heritage["lineage"] is reset to parameter.uid instead of its parent's
        """
        child = self.spawn_child()
        child.mutate()
        child.heritage["lineage"] = child.uid
        return child

    def recombine(self: P, *others: P) -> None:
        """Update value and parameters of this instance by combining it with
        other instances.

        Parameters
        ----------
        *others: Parameter
            other instances of the same type than this instance.
        """
        raise utils.NotSupportedError(f"Recombination is not implemented for {self.name}")

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
        assert reference is None or isinstance(reference, self.__class__), f"Expected {type(self)} but got {type(reference)} as reference"
        return self._internal_get_standardized_data(self if reference is None else reference)

    def _internal_get_standardized_data(self: P, reference: P) -> np.ndarray:
        raise utils.NotSupportedError(f"Export to standardized data space is not implemented for {self.name}")

    def set_standardized_data(self: P, data: tp.ArrayLike, *, reference: tp.Optional[P] = None, deterministic: bool = False) -> P:
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
        assert isinstance(sent_reference, self.__class__), f"Expected {type(self)} but got {type(sent_reference)} as reference"
        self._check_frozen()
        self._internal_set_standardized_data(np.array(data, copy=False), reference=sent_reference, deterministic=deterministic)
        return self

    def _internal_set_standardized_data(self: P, data: np.ndarray, reference: P, deterministic: bool = False) -> None:
        raise utils.NotSupportedError(f"Import from standardized data space is not implemented for {self.name}")

    # PART 2 - Additional features

    @property
    def generation(self) -> int:
        """Generation of the parameter (children are current generation + 1)
        """
        return self._generation

    def get_value_hash(self) -> tp.Hashable:
        """Hashable object representing the current value of the instance
        """
        val = self.value
        if isinstance(val, (str, bytes, float, int)):
            return val
        elif isinstance(val, np.ndarray):
            return val.tobytes()
        else:
            raise utils.NotSupportedError(f"Value hash is not supported for object {self.name}")

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
        substr = ""
        if self._parameters is not None and self.parameters:
            substr = f"[{self.parameters._get_parameters_str()}]"
            if substr == "[]":
                substr = ""
        return f"{self._get_name()}" + substr

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
        if self._parameters is not None and not self.parameters.satisfies_constraints():
            return False
        if not self._constraint_checkers:
            return True
        val = self.value
        return all(func(val) for func in self._constraint_checkers)

    def register_cheap_constraint(self, func: tp.Callable[[tp.Any], bool]) -> None:
        """Registers a new constraint on the parameter values.

        Parameters
        ----------
        func: Callable
            function which, given the value of the instance, returns whether it satisfies the constraints.

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
            seed = np.random.randint(2 ** 32, dtype=np.uint32)
            self._set_random_state(np.random.RandomState(seed))
        assert self._random_state is not None
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        self._set_random_state(random_state)

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        self._random_state = random_state
        if self._parameters is not None:
            self.parameters._set_random_state(random_state)

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
        rng = self.random_state  # make sure to create one before spawning
        child = self._internal_spawn_child()
        child._set_random_state(rng)
        child._constraint_checkers = list(self._constraint_checkers)
        child._generation = self.generation + 1
        child._descriptors = self._descriptors
        child._name = self._name
        child.parents_uids.append(self.uid)
        child.heritage = dict(self.heritage)
        if new_value is not None:
            child.value = new_value
        return child

    def freeze(self) -> None:
        """Prevents the parameter from changing value again (through value, mutate etc...)
        """
        self._frozen = True
        if self._parameters is not None:
            self._parameters.freeze()

    def _check_frozen(self) -> None:
        if self._frozen and not isinstance(self, Constant):  # nevermind constants (since they dont spawn children)
            raise RuntimeError(f"Cannot modify frozen Parameter {self}, please spawn a child and modify it instead"
                               "(optimizers freeze the parametrization and all asked and told candidates to avoid border effects)")

    def _internal_spawn_child(self: P) -> P:
        # default implem just forwards params
        inputs = {k: v.spawn_child() if isinstance(v, Parameter) else v for k, v in self.parameters._content.items()}
        child = self.__class__(**inputs)
        return child

    def copy(self: P) -> P:  # TODO test (see former instrumentation_copy test)
        """Create a child, but remove the random state
        This is used to run multiple experiments
        """
        child = self.spawn_child()
        child._name = self._name
        child.random_state = None
        return child

    def _compute_descriptors(self) -> utils.Descriptors:
        return utils.Descriptors()

    @property
    def descriptors(self) -> utils.Descriptors:
        if self._descriptors is None:
            self._compute_descriptors()
            self._descriptors = self._compute_descriptors()
        return self._descriptors


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
        if isinstance(value, Parameter):
            raise TypeError("Only non-parameters can be wrapped in a Constant")
        self._value = value

    def _get_name(self) -> str:
        return str(self._value)

    def get_value_hash(self) -> tp.Hashable:
        try:
            return super().get_value_hash()
        except utils.NotSupportedError:
            return "#non-hashable-constant#"

    @property
    def value(self) -> tp.Any:
        return self._value

    @value.setter
    def value(self, value: tp.Any) -> None:
        different = False
        if isinstance(value, np.ndarray):
            if not np.equal(value, self._value).all():
                different = True
        elif not (value == self._value or value is self._value):
            different = True
        if different:
            raise ValueError(f'Constant value can only be updated to the same value (in this case "{self._value}")')

    def get_standardized_data(self: P, *, reference: tp.Optional[P] = None) -> np.ndarray:  # pylint: disable=unused-argument
        return np.array([])

    # pylint: disable=unused-argument
    def set_standardized_data(self: P, data: tp.ArrayLike, *, reference: tp.Optional[P] = None, deterministic: bool = False) -> P:
        if np.array(data, copy=False).size:
            raise ValueError(f"Constant dimension should be 0 (got data: {data})")
        return self

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


class Dict(Parameter):
    """Dictionary-valued parameter. This Parameter can contain other Parameters,
    its value is a dict, with keys the ones provided as input, and corresponding values are
    either directly the provided values if they are not Parameter instances, or the value of those
    Parameters. It also implements a getter to access the Parameters directly if need be.

    Parameters
    ----------
    **parameters: Any
        the objects or Parameter which will provide values for the dict

    Note
    ----
    This is the base structure for all container Parameters, and it is
    used to hold the internal/model parameters for all Parameter classes.
    """

    def __init__(self, **parameters: tp.Any) -> None:
        super().__init__()
        self._content: tp.Dict[tp.Any, Parameter] = {k: as_parameter(p) for k, p in parameters.items()}
        self._sizes: tp.Optional[tp.Dict[str, int]] = None
        self._sanity_check(list(self._content.values()))
        self._ignore_in_repr: tp.Dict[str, str] = {}  # hacky undocumented way to bypass boring representations

    def _sanity_check(self, parameters: tp.List[Parameter]) -> None:
        """Check that all parameters are different
        """  # TODO: this is first order, in practice we would need to test all the different parameter levels together
        if parameters:
            assert all(isinstance(p, Parameter) for p in parameters)
            ids = {id(p) for p in parameters}
            if len(ids) != len(parameters):
                raise ValueError("Don't repeat twice the same parameter")

    def _compute_descriptors(self) -> utils.Descriptors:
        init = utils.Descriptors()
        return functools.reduce(operator.and_, [p.descriptors for p in self._content.values()], init)

    def __getitem__(self, name: tp.Any) -> Parameter:
        return self._content[name]

    def __len__(self) -> int:
        return len(self._content)

    def _get_parameters_str(self) -> str:
        params = sorted((k, p.name) for k, p in self._content.items()
                        if p.name != self._ignore_in_repr.get(k, "#ignoredrepr#"))
        return ",".join(f"{k}={n}" for k, n in params)

    def _get_name(self) -> str:
        return f"{self.__class__.__name__}({self._get_parameters_str()})"

    @property
    def value(self) -> tp.Dict[str, tp.Any]:
        return {k: as_parameter(p).value for k, p in self._content.items()}

    @value.setter
    def value(self, value: tp.Dict[str, tp.Any]) -> None:
        cls = self.__class__.__name__
        if not isinstance(value, dict):
            raise TypeError(f"{cls} value must be a dict, got: {value}\nCurrent value: {self.value}")
        if set(value) != set(self._content):
            raise ValueError(f"Got input keys {set(value)} for {cls} but expected {set(self._content)}\nCurrent value: {self.value}")
        for key, val in value.items():
            as_parameter(self._content[key]).value = val

    def get_value_hash(self) -> tp.Hashable:
        return tuple(sorted((x, y.get_value_hash()) for x, y in self._content.items()))

    def _internal_get_standardized_data(self: D, reference: D) -> np.ndarray:
        data = {k: self[k].get_standardized_data(reference=p) for k, p in reference._content.items()}
        if self._sizes is None:
            self._sizes = OrderedDict(sorted((x, y.size) for x, y in data.items()))
        assert self._sizes is not None
        data_list = [data[k] for k in self._sizes]
        if not data_list:
            return np.array([])
        return data_list[0] if len(data_list) == 1 else np.concatenate(data_list)  # type: ignore

    def _internal_set_standardized_data(self: D, data: np.ndarray, reference: D, deterministic: bool = False) -> None:
        if self._sizes is None:
            self.get_standardized_data(reference=self)
        assert self._sizes is not None
        if data.size != sum(v for v in self._sizes.values()):
            raise ValueError(f"Unexpected shape {data.shape} for {self} with dimension {self.dimension}:\n{data}")
        data = data.ravel()
        start, end = 0, 0
        for name, size in self._sizes.items():
            end = start + size
            self._content[name].set_standardized_data(data[start: end], reference=reference[name], deterministic=deterministic)
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"

    def mutate(self) -> None:
        # pylint: disable=pointless-statement
        self.random_state  # make sure to create one before using
        for param in self._content.values():
            param.mutate()

    def sample(self: D) -> D:
        child = self.spawn_child()
        child._content = {k: p.sample() for k, p in self._content.items()}
        child.heritage["lineage"] = child.uid
        return child

    def recombine(self, *others: "Dict") -> None:
        if not others:
            return
        # pylint: disable=pointless-statement
        self.random_state  # make sure to create one before using
        assert all(isinstance(o, self.__class__) for o in others)
        for k, param in self._content.items():
            param.recombine(*[o[k] for o in others])

    def _internal_spawn_child(self: D) -> D:
        child = self.__class__()
        child._content = {k: v.spawn_child() for k, v in self._content.items()}
        return child

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        super()._set_random_state(random_state)
        for param in self._content.values():
            if isinstance(param, Parameter):
                param._set_random_state(random_state)

    def satisfies_constraints(self) -> bool:
        compliant = super().satisfies_constraints()
        return compliant and all(param.satisfies_constraints() for param in self._content.values() if isinstance(param, Parameter))

    def freeze(self) -> None:
        super().freeze()
        for p in self._content.values():
            p.freeze()
