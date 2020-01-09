
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import warnings
from collections import OrderedDict
import typing as tp
import numpy as np
# pylint: disable=no-value-for-parameter


P = tp.TypeVar("P", bound="Parameter")
D = tp.TypeVar("D", bound="Dict")


class Descriptors(tp.NamedTuple):
    """Provides access to a set of descriptors for the parametrization
    This can be used within optimizers.
    """
    deterministic: bool = True
    continuous: bool = True


class NotSupportedError(RuntimeError):
    """This type of operation is not supported by the parameter.
    """


# pylint: disable=too-many-instance-attributes
class Parameter:
    """This provides the core functionality of a parameter, aka
    value, subparameters, mutation, recombination
    and additional features such as shared random state,
    constraint check, hashes, generation and naming.
    """

    def __init__(self, **subparameters: tp.Any) -> None:
        # Main features
        self.uid = uuid.uuid4().hex
        self.parents_uids: tp.List[str] = []
        self.heritage: tp.Dict[str, tp.Any] = {"lineage": self.uid}  # passed through to children
        self._subparameters = None if not subparameters else Dict(**subparameters)
        self._dimension: tp.Optional[int] = None
        # Additional convenient features
        self._random_state: tp.Optional[np.random.RandomState] = None  # lazy initialization
        self._generation = 0
        self._constraint_checkers: tp.List[tp.Callable[[tp.Any], bool]] = []
        self._name: tp.Optional[str] = None
        self._frozen = False
        self._meta: tp.Dict[str, tp.Any] = {}  # for anything algorithm related

    @property
    def descriptors(self) -> Descriptors:
        return Descriptors(deterministic=True, continuous=True)

    @property
    def value(self) -> tp.Any:
        raise NotImplementedError

    @value.setter
    def value(self, value: tp.Any) -> tp.Any:
        raise NotImplementedError

    @property
    def subparameters(self) -> "Dict":
        if self._subparameters is None:  # delayed instantiation to avoid infinte loop
            assert self.__class__ != Dict, "subparameters of Parameters dict should never be called"
            self._subparameters = Dict()
        assert self._subparameters is not None
        return self._subparameters

    @property
    def dimension(self) -> int:
        """Dimension of the standardized space for this parameter
        i.e size of the vector returned by get_standardized_data()
        """
        if self._dimension is None:
            try:
                self._dimension = self.get_standardized_data().size
            except NotSupportedError:
                self._dimension = 0
        return self._dimension

    def mutate(self) -> None:
        """Mutate subparameters of the instance, and then its value
        """
        self._check_frozen()
        self.subparameters.mutate()
        data = self.get_standardized_data()  # pylint: disable=assignment-from-no-return
        # let's assume the random state is already there (next class)
        self.set_standardized_data(data + self.random_state.normal(size=data.shape), deterministic=False)

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
        """Update value and subparameters of this instance by combining it with
        other instances.

        Parameters
        ----------
        *others: Parameter
            other instances of the same type than this instance.
        """
        raise NotSupportedError(f"Recombination is not implemented for {self.name}")

    def get_standardized_data(self: P, instance: tp.Optional[P] = None) -> np.ndarray:
        """Get the standardized data representing the value of the instance as an array in the optimization space.
        In this standardized space, a mutation is typically centered and reduced (sigma=1) Gaussian noise.
        The data only represent the value of this instance, not the subparameters (eg.: mutable sigma), hence it does not
        fully represent the state of the instance. Also, in stochastic cases, the value can be non-deterministically
        deduced from the data (eg.: categorical variable, for which data includes sampling weights for each value)

        Parameters
        ----------
        instance: Parameter
            the instance to represent in the standardized data space. By default this is "self", but other
            instances of the same type can be passed so as to be able to perform operations between them in the
            standardized data space (see note below)

        Returns
        -------
        np.ndarray
            the representation of the value in the optimization space

        Note
        ----
        Operations between different standardized data should only be performed if at least one of these conditions apply:
        - subparameters do not mutate (eg: sigma is constant)
        - each array was produced by the same instance in the exact same state (no mutation)
        """
        assert instance is None or isinstance(instance, self.__class__), f"Expected {type(self)} but got {type(instance)} as instance"
        return self._internal_get_standardized_data(self if instance is None else instance)

    def _internal_get_standardized_data(self: P, instance: P) -> np.ndarray:
        raise NotSupportedError(f"Export to standardized data space is not implemented for {self.name}")

    def set_standardized_data(self: P, data: np.ndarray, instance: tp.Optional[P] = None, deterministic: bool = False) -> P:
        """Updates the value of the provided instance (or self) using the standardized data.

        Parameters
        ----------
        np.ndarray
            the representation of the value in the optimization space
        instance: Parameter
            the instance to update ("self", if not provided)
        deterministic: bool
            whether the value should be deterministically drawn (max probability) in the case of stochastic parameters

        Returns
        -------
        Parameter
            the updated instance (self, or the provided instance)
        """
        assert isinstance(deterministic, bool)
        sent_instance = self if instance is None else instance
        assert isinstance(sent_instance, self.__class__), f"Expected {type(self)} but got {type(sent_instance)} as instance"
        sent_instance._check_frozen()
        return self._internal_set_standardized_data(data, sent_instance, deterministic=deterministic)

    def _internal_set_standardized_data(self: P, data: np.ndarray, instance: P, deterministic: bool = False) -> P:
        raise NotSupportedError(f"Import from standardized data space is not implemented for {self.name}")

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
            raise NotSupportedError(f"Value hash is not supported for object {self.name}")

    def get_data_hash(self) -> tp.Hashable:
        """Hashable object representing the current standardized data of the object.

        Note
        ----
        - this differs from the value hash, since the value is sometimes randomly sampled from the data
        - standardized data does not account for the full state of the instance (it does not contain
          data from subparameters)
        """
        return self.get_standardized_data().tobytes()

    def _get_name(self) -> str:
        """Internal implementation of parameter name. This should be value independant, and should not account
        for subparameters.
        """
        return self.__class__.__name__

    @property
    def name(self) -> str:
        """Name of the parameter
        This is used to keep track of how this Parameter is configured (included through subparameters),
        mostly for reproducibility A default version is always provided, but can be overriden directly
        through the attribute, or through the set_name method (which allows chaining).
        """
        if self._name is not None:
            return self._name
        substr = ""
        if self._subparameters is not None and self.subparameters:
            substr = f"[{self.subparameters._get_parameters_str()}]"
        return f"{self._get_name()}" + substr

    @name.setter
    def name(self, name: str) -> None:
        self.set_name(name)  # with_name allows chaining

    def __repr__(self) -> str:
        return f"{self.name}:{self.value}".replace(" ", "").replace("\n", "")

    def set_name(self: P, name: str) -> P:
        """Sets a name and return the current instrumentation (for chaining)

        Parameter
        ---------
        name: str
            new name to use to represent the Parameter
        """
        self._name = name
        return self

    # %% Constraint management

    def satisfies_constraint(self) -> bool:
        """Whether the instance satisfies the constraints added through
        the "register_cheap_constraint" method

        Returns
        -------
        bool
            True iff the constraint is satisfied
        """
        if self._subparameters is not None and not self.subparameters.satisfies_constraint():
            return False
        if not self._constraint_checkers:
            return True
        val = self.value
        return all(func(val) for func in self._constraint_checkers)

    def register_cheap_constraint(self, func: tp.Callable[[tp.Any], bool]) -> None:
        """Registers a new constraint on the parameter values.

        Parameter
        ---------
        func: Callable
            function which, given the value of the instance, returns whether it satisfies the constraintp.

        Note
        - this is only for checking after mutation/recombination/etc if the value still satisfy the constraints.
          The constraint is not used in those processes.
        - constraints should be fast to compute.
        """
        if getattr(func, "__name__", "not lambda") == "<lambda>":  # LambdaType does not work :(
            warnings.warn("Lambda as constraint is not advice because it may not be picklable")
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
        if self._subparameters is not None:
            self.subparameters._set_random_state(random_state)

    def spawn_child(self: P, new_value: tp.Optional[tp.Any] = None) -> P:
        """Creates a new instance which shares the same random generator than its parent,
        is sampled from the same data, and mutates independently from the parentp.
        If a new value is provided, it will be set to the new instance

        Parameter
        ---------
        value: anything (optional)
            if provided, it will be set to the new instance.

        Returns
        -------
        Parameter
            a new instance of the same class, with same parameters/subparameters/...
            Optionally, a new value will be set after creation
        """
        rng = self.random_state  # make sure to create one before spawning
        child = self._internal_spawn_child()
        child._set_random_state(rng)
        child._constraint_checkers = list(self._constraint_checkers)
        child._generation = self.generation + 1
        child.parents_uids.append(self.uid)
        child.heritage = dict(self.heritage)
        if new_value is not None:
            child.value = new_value
        return child

    def freeze(self) -> None:
        """Prevents the parameter from changing value again (through value, mutate etc...)
        """
        self._frozen = True
        if self._subparameters is not None:
            self._subparameters.freeze()

    def _check_frozen(self) -> None:
        if self._frozen and not isinstance(self, Constant):  # nevermind constants (since they dont spawn children)
            raise RuntimeError(f"Cannot modify frozen Parameter {self}, please spawn a child and modify it instead")

    def _internal_spawn_child(self: P) -> P:
        # default implem just forwards params
        inputs = {k: v.spawn_child() if isinstance(v, Parameter) else v for k, v in self.subparameters._parameters.items()}
        child = self.__class__(**inputs)
        return child


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
        except NotSupportedError:
            return "#non-hashable-constant#"

    @property
    def value(self) -> tp.Any:
        return self._value

    @value.setter
    def value(self, value: tp.Any) -> None:
        if not value == self._value:
            raise ValueError(f'Constant value can only be updated to the same value (in this case "{self._value}")')

    def _internal_get_standardized_data(self: P, instance: P) -> np.ndarray:
        return np.array([])

    def _internal_set_standardized_data(self: P, data: np.ndarray, instance: P, deterministic: bool = False) -> P:
        if data.size:
            raise ValueError(f"Constant dimension should be 0 (got data: {data})")
        return instance

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
    used to hold the subparameters for all Parameter classes.
    """

    def __init__(self, **parameters: tp.Any) -> None:
        super().__init__()
        self._parameters: tp.Dict[tp.Any, Parameter] = {k: as_parameter(p) for k, p in parameters.items()}
        self._sizes: tp.Optional[tp.Dict[str, int]] = None
        self._sanity_check(list(parameters.values()))

    def _sanity_check(self, parameters: tp.List[Parameter]) -> None:
        """Check that all subparameters are different
        """  # TODO: this is first order, in practice we would need to test all the different parameter levels together
        ids = {id(p) for p in parameters}
        if len(ids) != len(parameters):
            raise ValueError("Don't repeat twice the same parameter")

    @property
    def descriptors(self) -> Descriptors:
        return Descriptors(**{name: all(getattr(as_parameter(p).descriptors, name) for p in self._parameters.values())
                              for name in ("deterministic", "continuous")})

    def __getitem__(self, name: tp.Any) -> Parameter:
        return self._parameters[name]

    def __len__(self) -> int:
        return len(self._parameters)

    def _get_parameters_str(self) -> str:
        params = sorted((k, p.name) for k, p in self._parameters.items())
        return ",".join(f"{k}={n}" for k, n in params)

    def _get_name(self) -> str:
        return f"{self.__class__.__name__}({self._get_parameters_str()})"

    @property
    def value(self) -> tp.Dict[str, tp.Any]:
        return {k: as_parameter(p).value for k, p in self._parameters.items()}

    @value.setter
    def value(self, value: tp.Dict[str, tp.Any]) -> None:
        if set(value) != set(self._parameters):
            raise ValueError(f"Got input keys {set(value)} but expected {set(self._parameters)}")
        for key, val in value.items():
            as_parameter(self._parameters[key]).value = val

    def get_value_hash(self) -> tp.Hashable:
        return tuple(sorted((x, y.get_value_hash()) for x, y in self._parameters.items()))

    def _internal_get_standardized_data(self: D, instance: D) -> np.ndarray:
        data = {k: self[k].get_standardized_data(p) for k, p in instance._parameters.items()}
        if self._sizes is None:
            self._sizes = OrderedDict(sorted((x, y.size) for x, y in data.items()))
        assert self._sizes is not None
        data_list = [data[k] for k in self._sizes]
        if not data_list:
            return np.array([])
        return data_list[0] if len(data_list) == 1 else np.concatenate(data_list)  # type: ignore

    def _internal_set_standardized_data(self: D, data: np.ndarray, instance: D, deterministic: bool = False) -> D:
        if self._sizes is None:
            self.get_standardized_data()
        assert self._sizes is not None
        assert data.size == sum(v for v in self._sizes.values())
        data = data.ravel()
        start, end = 0, 0
        for name, size in self._sizes.items():
            end = start + size
            self._parameters[name].set_standardized_data(data[start: end], instance=instance[name], deterministic=deterministic)
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"
        return instance

    def mutate(self) -> None:
        for param in self._parameters.values():
            param.mutate()

    def sample(self: D) -> D:
        child = self.spawn_child()
        child._parameters = {k: p.sample() for k, p in self._parameters.items()}
        child.heritage["lineage"] = child.uid
        return child

    def recombine(self, *others: "Dict") -> None:
        assert all(isinstance(o, self.__class__) for o in others)
        for k, param in self._parameters.items():
            param.recombine(*[o[k] for o in others])

    def _internal_spawn_child(self: D) -> D:
        child = self.__class__()
        child._parameters = {k: v.spawn_child() for k, v in self._parameters.items()}
        return child

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        super()._set_random_state(random_state)
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                param._set_random_state(random_state)

    def satisfies_constraint(self) -> bool:
        compliant = super().satisfies_constraint()
        return compliant and all(param.satisfies_constraint() for param in self._parameters.values() if isinstance(param, Parameter))

    def freeze(self) -> None:
        super().freeze()
        for p in self._parameters.values():
            p.freeze()
