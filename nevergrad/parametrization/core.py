# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import warnings
from collections import OrderedDict
import typing as t
import numpy as np


BP = t.TypeVar("BP", bound="BaseParameter")
P = t.TypeVar("P", bound="Parameter")
D = t.TypeVar("D", bound="Dict")


class NotSupportedError(RuntimeError):
    """This type of operation is not supported by the parameter.
    """


# pylint: disable=too-many-instance-attributes
class BaseParameter:
    """This provides the core functionality of a parameter, aka
    value, subparameters, mutation, recombination
    """

    def __init__(self, **subparameters: t.Any) -> None:
        self.uid = uuid.uuid4().hex
        self.parents_uids: t.List[str] = []
        self._subparameters = None if not subparameters else Dict(**subparameters)
        self._dimension: t.Optional[int] = None

    @property
    def value(self) -> t.Any:
        raise NotImplementedError

    @value.setter
    def value(self, value: t.Any) -> t.Any:
        raise NotImplementedError

    def spawn_child(self: BP) -> BP:
        raise NotImplementedError

    @property
    def subparameters(self) -> "Dict":
        if self._subparameters is None:  # delayed instantiation to avoid infinte loop
            assert self.__class__ != Dict, "subparameters of Parameters dict should never be called"
            self._subparameters = Dict()
        assert self._subparameters is not None
        return self._subparameters

    def _get_parameter_value(self, name: str) -> t.Any:
        param = self.subparameters[name]
        return param.value if isinstance(param, Parameter) else param

    @property
    def dimension(self) -> int:
        """Dimension of the standardized space for this parameter
        i.e size of the vector returned by get_std_data()
        """
        if self._dimension is None:
            try:
                self._dimension = self.get_std_data().size
            except NotSupportedError:
                self._dimension = 0
        return self._dimension

    def mutate(self) -> None:
        """Mutate subparameters of the instance, and then its value
        """
        self.subparameters.mutate()
        data = self.get_std_data()  # pylint: disable=assignment-from-no-return
        # let's assume the random state is already there (next class)
        self.set_std_data(data + self.random_state.normal(size=data.shape), deterministic=False)  # type: ignore

    def sample(self: BP) -> BP:
        """Sample a new instance of the parameter.
        This usually means spawning a child and mutating it.
        """
        child = self.spawn_child()
        child.mutate()
        return child

    def recombine(self: BP, *others: BP) -> None:
        """Update value and subparameters of this instance by combining it with
        other instances.

        Parameters
        ----------
        *others: Parameter
            other instances of the same type than this instance.
        """
        raise NotSupportedError(f"Recombination is not implemented for {self.name}")  # type: ignore

    def get_std_data(self: BP, instance: t.Optional[BP] = None) -> np.ndarray:
        assert instance is None or isinstance(instance, self.__class__), f"Expected {type(self)} but got {type(instance)} as instance"
        return self._internal_get_std_data(self if instance is None else instance)

    def _internal_get_std_data(self: BP, instance: BP) -> np.ndarray:
        raise NotSupportedError(f"Export to standardized data space is not implemented for {self.name}")  # type: ignore

    def set_std_data(self: BP, data: np.ndarray, instance: t.Optional[BP] = None, deterministic: bool = False) -> BP:
        assert isinstance(deterministic, bool)
        assert instance is None or isinstance(instance, self.__class__), f"Expected {type(self)} but got {type(instance)} as instance"
        return self._internal_set_std_data(data, instance=self if instance is None else instance, deterministic=deterministic)

    def _internal_set_std_data(self: BP, data: np.ndarray, instance: BP, deterministic: bool = False) -> BP:
        raise NotSupportedError(f"Import from standardized data space is not implemented for {self.name}")  # type: ignore

    def from_value(self: BP, value: t.Any) -> BP:
        child = self.spawn_child()
        child.value = value
        return child


# pylint: disable=abstract-method
class Parameter(BaseParameter):
    """This provides the core functionality of a parameter, aka
    value, subparameters, mutation, recombination
    and adds some additional features such as shared random state,
    constraint check, hashes, generation and naming.
    """

    def __init__(self, **subparameters: t.Any) -> None:
        super().__init__(**subparameters)
        self._random_state: t.Optional[np.random.RandomState] = None  # lazy initialization
        self._generation = 0
        self._constraint_checkers: t.List[t.Callable[[t.Any], bool]] = []
        self._name: t.Optional[str] = None

    @property
    def generation(self) -> int:
        """generation of the parameter (children are current generation + 1)
        """
        return self._generation

    def get_value_hash(self) -> t.Hashable:
        """Hashable object representing the current value of the instance
        """
        val = self.value
        if isinstance(val, (str, bytes, float, int)):
            return val
        elif isinstance(val, np.ndarray):
            return val.tobytes()
        else:
            raise NotSupportedError(f"Value hash is not supported for object {self.name}")

    def get_data_hash(self) -> t.Hashable:
        """Hashable object representing the current standardized data of the object.

        Note
        ----
        - this differs from the value hash, since the value is sometimes randomly sampled from the data
        - standardized data does not account for the full state of the instance (it does not contain
          data from subparameters)
        """
        return self.get_std_data().tobytes()

    def _get_name(self) -> str:
        return self.__class__.__name__

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        substr = ""
        if self._subparameters is not None:
            subparams = sorted((k, p.name if isinstance(p, Parameter) else p) for k, p in self.subparameters.value.items())
            if subparams:
                subparams = "[" + ",".join(f"{k}={n}" for k, n in subparams) + "]"  # type:ignore
        return f"{self._get_name()}" + substr

    @name.setter
    def name(self, name: str) -> None:
        self.set_name(name)  # with_name allows chaining

    def __repr__(self) -> str:
        return f"{self.name}:{self.value}".replace(" ", "").replace("\n", "")

    def set_name(self: P, name: str) -> P:
        """Sets a name and return the current instrumentation (for chaining)
        """
        self._name = name
        return self

    # %% Constraint management

    def satisfies_constraint(self) -> bool:
        """Whether the instance complies with the constraints added through
        the "register_cheap_constraint" method
        """
        if not self._constraint_checkers:
            return True
        val = self.value
        return all(func(val) for func in self._constraint_checkers)

    def register_cheap_constraint(self, func: t.Callable[[t.Any], bool]) -> None:
        """Registers a new constraint on the parameter values.

        Parameter
        ---------
        func: Callable
            function which, given the value of the instance, returns whether it satisfies the constraint.

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

    def spawn_child(self: P) -> P:
        rng = self.random_state  # make sure to create one before spawning
        child = self._internal_spawn_child()
        child._set_random_state(rng)
        child._constraint_checkers = list(self._constraint_checkers)
        child._generation = self.generation + 1
        child.parents_uids.append(self.uid)
        return child

    def _internal_spawn_child(self: P) -> P:
        # default implem just forwards params
        inputs = {k: v.spawn_child() if isinstance(v, Parameter) else v for k, v in self.subparameters._parameters.items()}
        child = self.__class__(**inputs)
        return child


class Constant(Parameter):

    def __init__(self, value: t.Any) -> None:
        super().__init__()
        if isinstance(value, Parameter):
            raise TypeError("Only non-parameters can be wrapped in a Constant")
        self._value = value

    def _get_name(self) -> str:
        return str(self._value)

    @property
    def value(self) -> t.Any:
        return self._value

    @value.setter
    def value(self, value: t.Any) -> None:
        if not value == self._value:
            raise ValueError(f'Constant value can only be updated to the same value (in this case "{self._value}")')

    def _internal_get_std_data(self: BP, instance: BP) -> np.ndarray:
        return np.array([])

    def _internal_set_std_data(self: P, data: np.ndarray, instance: P, deterministic: bool = False) -> P:
        if data.size:
            raise ValueError(f"Constant dimension should be 0 (got data: {data})")
        return instance

    def spawn_child(self: P) -> P:
        return self  # no need to create another instance for a constant


def _as_parameter(param: t.Any) -> Parameter:
    if isinstance(param, Parameter):
        return param
    else:
        return Constant(param)


class Dict(Parameter):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, **parameters: t.Any) -> None:
        super().__init__()
        self._parameters: t.Dict[t.Any, t.Any] = parameters
        self._sizes: t.Optional[t.Dict[str, int]] = None

    def __getitem__(self, name: t.Any) -> t.Any:
        return self._parameters[name]

    def _get_name(self) -> str:
        params = sorted((k, p.name if isinstance(p, Parameter) else p) for k, p in self._parameters.items())
        paramsstr = "{" + ",".join(f"{k}={n}" for k, n in params) + "}"
        return f"{self.__class__.__name__}{paramsstr}"

    @property
    def value(self) -> t.Dict[str, t.Any]:
        return {k: _as_parameter(p).value for k, p in self._parameters.items()}

    @value.setter
    def value(self, value: t.Dict[str, t.Any]) -> None:
        if set(value) != set(self._parameters):
            raise ValueError(f"Got input keys {set(value)} but expected {set(self._parameters)}")
        for key, val in value.items():
            _as_parameter(self._parameters[key]).value = val

    def get_value_hash(self) -> t.Hashable:
        return tuple(sorted((x, y.get_value_hash()) for x, y in self._parameters.items() if isinstance(y, Parameter)))

    def _internal_get_std_data(self: D, instance: D) -> np.ndarray:
        data = {k: self[k].get_std_data(p) for k, p in instance._parameters.items() if isinstance(p, Parameter)}
        if self._sizes is None:
            self._sizes = OrderedDict(sorted((x, y.size) for x, y in data.items()))
        assert self._sizes is not None
        data_list = [data[k] for k in self._sizes]
        if not data_list:
            return np.array([])
        return data_list[0] if len(data_list) == 1 else np.concatenate(data_list)  # type: ignore

    def _internal_set_std_data(self: D, data: np.ndarray, instance: D, deterministic: bool = False) -> D:
        if self._sizes is None:
            self.get_std_data()
        assert self._sizes is not None
        assert data.size == sum(v for v in self._sizes.values())
        data = data.ravel()
        start, end = 0, 0
        for name, size in self._sizes.items():
            end = start + size
            self._parameters[name].set_std_data(data[start: end], instance=instance[name], deterministic=deterministic)
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"
        return instance

    def mutate(self) -> None:
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                param.mutate()

    def recombine(self, *others: "Dict") -> None:
        assert all(isinstance(o, self.__class__) for o in others)
        for k, param in self._parameters.items():
            if isinstance(param, Parameter):
                param.recombine(*[o[k] for o in others])

    def _internal_spawn_child(self: D) -> D:
        child = self.__class__()
        child._parameters = {k: v.spawn_child() if isinstance(v, Parameter) else v for k, v in self._parameters.items()}
        return child

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        super()._set_random_state(random_state)
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                param._set_random_state(random_state)

    def satisfies_constraint(self) -> bool:
        compliant = super().satisfies_constraint()
        return compliant and all(param.satisfies_constraint() for param in self._parameters.values() if isinstance(param, Parameter))
