import uuid
from collections import OrderedDict
from typing import TypeVar, List, Optional, Dict, Any, Callable
import numpy as np


BP = TypeVar("BP", bound="BaseParameter")
P = TypeVar("P", bound="Parameter")


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
        self._subparameters = None if not subparameters else ParametersDict(**subparameters)
        self._dimension: Optional[int] = None

    @property
    def value(self) -> Any:
        raise NotImplementedError

    @property
    def subparameters(self) -> "ParametersDict":
        if self._subparameters is None:  # delayed instantiation to avoid infinte loop
            self._subparameters = ParametersDict()
        assert self._subparameters is not None
        return self._subparameters

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            try:
                self._dimension = self.to_std_data().size
            except NotSupportedError:
                self._dimension = 0
        return self._dimension

    def mutate(self) -> None:
        self.subparameters.mutate()
        data = self.to_std_data()  # pylint: disable=assignment-from-no-return
        self.with_std_data(data + np.random.normal(size=data.shape))

    def sample(self: BP) -> BP:
        child = self.spawn_child()
        child.mutate()
        return child

    def recombine(self: BP, *others: BP) -> None:
        raise NotSupportedError

    def to_std_data(self) -> np.ndarray:
        raise NotSupportedError

    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        raise NotSupportedError

    def spawn_child(self: BP) -> BP:
        raise NotImplementedError

    def from_value(self: BP, value: Any) -> BP:
        child = self.spawn_child()
        child.value = value  # type: ignore
        return child


# pylint: disable=abstract-method
class Parameter(BaseParameter):
    """This provides the core functionality of a parameter, aka
    value, subparameters, mutation, recombination
    and adds some additional features such as shared random state,
    constraint check and naming.
    """

    def __init__(self, **subparameters: Any) -> None:
        super().__init__(**subparameters)
        self._random_state: Optional[np.random.RandomState] = None  # lazy initialization
        self._constraint_checker: Optional[Callable[[Any], bool]] = None
        self._name: Optional[str] = None

    def _get_name(self) -> str:
        return self.__class__.__name__

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name
        subparams = sorted((k, p.name if isinstance(p, Parameter) else p) for k, p in self.subparameters.value.items())
        substr = ""
        if subparams:
            subparams = "[" + ",".join(f"{k}={n}" for k, n in subparams) + "]"  # type:ignore
        return f"{self._get_name()}" + substr

    def __repr__(self) -> str:
        return f"{self.name}:{self.value}".replace(" ", "").replace("\n", "")

    def with_name(self: P, name: str) -> P:
        """Sets a name and return the current instrumentation (for chaining)
        """
        self._name = name
        return self

    # %% Constraint management

    def complies_with_constraint(self) -> bool:
        if self._constraint_checker is None:
            return True
        else:
            return self._constraint_checker(self.value)

    def set_cheap_constraint_checker(self, func: Callable[[Any], bool]) -> None:
        self._constraint_checker = func

    # %% random state

    @property
    def random_state(self) -> np.random.RandomState:
        """Random state the instrumentation and the optimizers pull from.
        It can be seeded/replaced.
        """
        if self._random_state is None:
            # use the setter, to make sure the random state is propagated to the variables
            seed = np.random.randint(2 ** 32, dtype=np.uint32)
            self.random_state = np.random.RandomState(seed)
        assert self._random_state is not None
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        self._set_random_state(random_state)

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        self._random_state = random_state
        if self._subparameters is not None:
            self.subparameters.random_state = random_state


class ParametersDict(Parameter):
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

    def to_std_data(self) -> np.ndarray:
        data = {k: p.to_std_data() for k, p in self._parameters.items() if isinstance(p, Parameter)}
        if self._sizes is None:
            self._sizes = OrderedDict(sorted((x, y.size) for x, y in data.items()))
        assert self._sizes is not None
        return np.concatenate([data[k] for k in self._sizes])  # type: ignore

    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        if self._sizes is None:
            self.to_std_data()
        assert self._sizes is not None
        assert data.size == sum(v for v in self._sizes.values())
        data = data.ravel()
        start, end = 0, 0
        for name, size in self._sizes.items():
            end = start + size
            self._parameters[name].with_std_data(data[start: end], deterministic)
            start = end
        assert end == len(data), f"Finished at {end} but expected {len(data)}"

    def mutate(self) -> None:
        for param in self._parameters.values():
            if isinstance(param, Parameter):
                param.mutate()

    def recombine(self, *others: "ParametersDict") -> None:
        for k, param in self._parameters.items():
            param.recombine([o._parameters[k] for o in others])

    def spawn_child(self) -> "ParametersDict":
        child = ParametersDict(**{k: v.spawn_child() for k, v in self._parameters.items()})
        child.parents_uids.append(self.uid)
        return child

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        self._random_state = random_state
        for param in self._parameters.values():
            param.random_state = random_state

    def complies_with_constraint(self) -> bool:
        compliant = super().complies_with_constraint()
        return compliant and all(param.complies_to_constraint() for param in self._parameters.values())


class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self._value: np.ndarray = np.zeros(dims)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, new_value: np.ndarray) -> None:
        if not isinstance(new_value, np.ndarray):
            raise TypeError(f"Received a {type(new_value)} in place of a np.ndarray")
        if self._value.shape != new_value.shape:
            raise ValueError(f"Cannot set array of shape {self._value.shape} with value of shape {new_value.shape}")
        self._value = new_value

    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        self._value = data.reshape(self.value.shape)

    def to_std_data(self) -> np.ndarray:
        return self._value.ravel()

    def spawn_child(self) -> "Array":
        child = Array(*self.value.shape)
        child._value = self.value
        child.parents_uids.append(self.uid)
        return child
