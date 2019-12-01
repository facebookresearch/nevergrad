import uuid
from collections import OrderedDict
from typing import TypeVar, List, Optional, Dict, Any, Callable, Union, Tuple
import numpy as np


BP = TypeVar("BP", bound="BaseParameter")
P = TypeVar("P", bound="Parameter")
D = TypeVar("D", bound="NgDict")


class NotSupportedError(RuntimeError):
    """This type of operation is not supported by the parameter.
    """


# pylint: disable=too-many-instance-attributes
class BaseParameter:
    """This provides the core functionality of a parameter, aka
    value, subparameters, mutation, recombination
    """

    def __init__(self, **subparameters: Any) -> None:
        self.uid = uuid.uuid4().hex
        self.parents_uids: List[str] = []
        self._subparameters = None if not subparameters else NgDict(**subparameters)
        self._dimension: Optional[int] = None

    @property
    def value(self) -> Any:
        raise NotImplementedError

    @value.setter
    def value(self, value: Any) -> Any:
        raise NotImplementedError

    def spawn_child(self: BP) -> BP:
        inputs = {k: v.spawn_child() if isinstance(v, Parameter) else v for k, v in self.subparameters._parameters.items()}
        child = self.__class__(**inputs)
        child.parents_uids.append(self.uid)
        return child

    @property
    def subparameters(self) -> "NgDict":
        if self._subparameters is None:  # delayed instantiation to avoid infinte loop
            assert self.__class__ != NgDict, "subparameters of Parameters dict should never be called"
            self._subparameters = NgDict()
        assert self._subparameters is not None
        return self._subparameters

    def _get_parameter_value(self, name: str) -> Any:
        param = self.subparameters._parameters[name]
        return param.value if isinstance(param, Parameter) else param

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            try:
                self._dimension = self.get_std_data().size
            except NotSupportedError:
                self._dimension = 0
        return self._dimension

    def mutate(self) -> None:
        self.subparameters.mutate()
        data = self.get_std_data()  # pylint: disable=assignment-from-no-return
        # let's assume the random state is already there (next class)
        self.set_std_data(data + self.random_state.normal(size=data.shape))  # type: ignore

    def sample(self: BP) -> BP:
        child = self.spawn_child()
        child.mutate()
        return child

    def recombine(self: BP, *others: BP) -> None:
        raise NotSupportedError(f"Recombination is not implemented for {self.name}")  # type: ignore

    def get_std_data(self) -> np.ndarray:
        raise NotSupportedError(f"Export to standardized data space is not implemented for {self.name}")  # type: ignore

    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        raise NotSupportedError(f"Import from standardized data space is not implemented for {self.name}")  # type: ignore

    def from_value(self: BP, value: Any) -> BP:
        child = self.spawn_child()
        child.value = value
        return child


# pylint: disable=abstract-method
class Parameter(BaseParameter):
    """This provides the core functionality of a parameter, aka
    value, subparameters, mutation, recombination
    and adds some additional features such as shared random state,
    constraint check, hashes and naming.
    """

    def __init__(self, **subparameters: Any) -> None:
        super().__init__(**subparameters)
        self._random_state: Optional[np.random.RandomState] = None  # lazy initialization
        self._constraint_checkers: List[Callable[[Any], bool]] = []
        self._name: Optional[str] = None

    def compute_value_hash(self) -> Any:
        val = self.value
        if isinstance(val, (str, bytes, float, int)):
            return val
        elif isinstance(val, np.ndarray):
            return val.tobytes()
        else:
            raise NotSupportedError(f"Value hash is not supported for object {self.name}")

    def compute_data_hash(self) -> Union[str, bytes, float, int]:
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

    def complies_with_constraint(self) -> bool:
        if not self._constraint_checkers:
            return True
        val = self.value
        return all(func(val) for func in self._constraint_checkers)

    def register_cheap_constraint(self, func: Callable[[Any], bool]) -> None:
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
        return child

    def _internal_spawn_child(self: P) -> P:
        return super().spawn_child()


class NgDict(Parameter):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, **parameters: Any) -> None:
        super().__init__()
        self._parameters = parameters
        self._sizes: Optional[Dict[str, int]] = None

    def _get_name(self) -> str:
        params = sorted((k, p.name if isinstance(p, Parameter) else p) for k, p in self._parameters.items())
        paramsstr = "{" + ",".join(f"{k}={n}" for k, n in params) + "}"
        return f"{self.__class__.__name__}{paramsstr}"

    @property
    def value(self) -> Dict[str, Any]:
        return {k: p.value if isinstance(p, Parameter) else p for k, p in self._parameters.items()}

    @value.setter
    def value(self, value: Dict[str, Any]) -> None:
        if set(value) != set(self._parameters):
            raise ValueError(f"Got input keys {set(value)} but expected {set(self._parameters)}")
        for key, val in value.items():
            param = self._parameters[key]
            if isinstance(param, Parameter):
                param.value = val
            else:
                if not param == val:  # safer this way
                    raise ValueError(f"Trying to set frozen value {key}={param} to {val}")  # TODO test this

    def compute_value_hash(self) -> Tuple[Tuple[str, Any], ...]:
        return tuple(sorted((x, y.compute_value_hash()) for x, y in self._parameters.items() if isinstance(y, Parameter)))

    def get_std_data(self) -> np.ndarray:
        data = {k: p.get_std_data() for k, p in self._parameters.items() if isinstance(p, Parameter)}
        if self._sizes is None:
            self._sizes = OrderedDict(sorted((x, y.size) for x, y in data.items()))
        assert self._sizes is not None
        data_list = [data[k] for k in self._sizes]
        if not data_list:
            return np.array([])
        return data_list[0] if len(data_list) == 1 else np.concatenate(data_list)  # type: ignore

    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        if self._sizes is None:
            self.get_std_data()
        assert self._sizes is not None
        assert data.size == sum(v for v in self._sizes.values())
        data = data.ravel()
        start, end = 0, 0
        for name, size in self._sizes.items():
            end = start + size
            self._parameters[name].set_std_data(data[start: end], deterministic)
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"

    def mutate(self) -> None:
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                param.mutate()

    def recombine(self, *others: "NgDict") -> None:
        for k, param in self._parameters.items():
            if isinstance(param, Parameter):
                param.recombine(*[o._parameters[k] for o in others])

    def _internal_spawn_child(self: D) -> D:
        child = self.__class__()
        child._parameters = {k: v.spawn_child() if isinstance(v, Parameter) else v for k, v in self._parameters.items()}
        child.parents_uids.append(self.uid)
        return child

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        super()._set_random_state(random_state)
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                param._set_random_state(random_state)

    def complies_with_constraint(self) -> bool:
        compliant = super().complies_with_constraint()
        return compliant and all(param.complies_with_constraint() for param in self._parameters.values() if isinstance(param, Parameter))
