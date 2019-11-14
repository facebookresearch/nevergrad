import uuid
from collections import OrderedDict
from typing import TypeVar, List, Optional, Dict, Any
import numpy as np


P = TypeVar("P", bound="Parameter")


class NotSupportedError(RuntimeError):
    pass


class Parameter:

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

    def recombine(self: P, *others: P) -> None:
        raise NotSupportedError

    def to_std_data(self) -> np.ndarray:
        raise NotSupportedError

    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        raise NotSupportedError

    def spawn_child(self: P) -> P:
        raise NotImplementedError

    def from_value(self: P, value: Any) -> P:
        child = self.spawn_child()
        child.value = value  # type: ignore
        return child


class ParametersDict(Parameter):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, **parameters: Any) -> None:
        super().__init__()
        self._parameters = parameters
        self._sizes: Optional[Dict[str, int]] = None

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


class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self._value: np.ndarray = np.zeros(dims)

    @property
    def value(self) -> np.ndarray:
        return self._value

    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        self._value = data.reshape(self.value.shape)

    def to_std_data(self) -> np.ndarray:
        return self._value.ravel()

    def spawn_child(self) -> "Array":
        child = Array(*self.value.shape)
        child._value = self.value
        child.parents_uids.append(self.uid)
        return child
