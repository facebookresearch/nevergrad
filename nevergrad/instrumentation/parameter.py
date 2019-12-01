from typing import Union, Tuple, Any, List, Iterable
import numpy as np
# importing NgDict to populate parameters (fake renaming for mypy explicit reimport)
# pylint: disable=unused-import,useless-import-alias
from . import discretization
from .core3 import Parameter
from .core3 import NgDict as NgDict  # noqa


class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.

    Parameters
    ----------
    sigma: float or Array
        standard deviation of a mutation
    distribution: str
        distribution of the data ("linear" or "log")
    """

    def __init__(
            self,
            shape: Tuple[int, ...],
            sigma: Union[float, "Array"] = 1.0,
            distribution: Union[str, Parameter] = "linear",
            recombination: Union[str, Parameter] = "average"
    ) -> None:
        assert not isinstance(shape, Parameter)
        super().__init__(shape=shape, sigma=sigma, distribution=distribution, recombination=recombination)
        self._value: np.ndarray = np.zeros(shape)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray")
        if self._value.shape != value.shape:
            raise ValueError(f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}")
        self._value = value

    # pylint: disable=unused-argument
    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        sigma = self._get_parameter_value("sigma")
        self._value = (sigma * data).reshape(self.value.shape)

    def spawn_child(self) -> "Array":
        child = super().spawn_child()
        child._value = self.value
        return child

    def get_std_data(self) -> np.ndarray:
        sigma = self._get_parameter_value("sigma")
        reduced = self._value / sigma
        return reduced.ravel()  # type: ignore

    def recombine(self, *others: "Array") -> None:
        recomb = self._get_parameter_value("recombination")
        all_p = [self] + list(others)
        if recomb == "average":
            self.set_std_data(np.mean([p.get_std_data() for p in all_p], axis=0))
        else:
            raise ValueError(f'Unknown recombination "{recomb}"')


class NgList(NgDict):
    """Handle for facilitating dict of parameters management
    """

    def __init__(self, *parameters: Any) -> None:
        super().__init__(**{str(k): p for k, p in enumerate(parameters)})

    @property  # type: ignore
    def value(self) -> List[Any]:  # type: ignore
        param_val = [x[1] for x in sorted(self._parameters.items(), key=lambda x: int(x[0]))]
        return [p.value if isinstance(p, Parameter) else p for p in param_val]

    @value.setter
    def value(self, value: List[Any]) -> None:
        assert isinstance(value, list)
        for k, val in enumerate(value):
            key = str(k)
            param = self._parameters[key]
            if not isinstance(param, Parameter):
                self._parameters[key] = val
            else:
                param.value = val


class Choice(Parameter):

    def __init__(
            self,
            choices: Iterable[Any],
            recombination: Union[str, Parameter] = "average",
            deterministic: bool = False,
    ) -> None:
        assert not isinstance(choices, NgList)
        lchoices = list(choices)  # for iterables
        super().__init__(probabilities=Array(shape=(len(lchoices),), recombination=recombination),
                         choices=NgList(*lchoices))
        self._deterministic = deterministic
        self._index = 0
        self._draw(deterministic=False)

    @property
    def value(self) -> Any:
        val = self.subparameters._parameters["choices"][str(self._index)]
        return val.value if isinstance(val, Parameter) else val

    @value.setter
    def value(self, value: Any) -> None:
        index = -1
        # try to find where to put this
        choices = self.subparameters._parameters["choices"]
        nums = sorted(choices)
        for k in nums:
            choice = choices[k]
            if isinstance(choice, Parameter):
                try:
                    choice.value = value
                except Exception:  # pylint: disable=broad-except
                    pass
                else:
                    index = int(k)
                    break
            else:
                if not value != choice:  # slighly safer this way
                    index = int(k)
                    break
        if index == -1:
            raise ValueError(f"Could not figure out where to put value {value}")
        out = discretization.inverse_softmax_discretization(index, len(nums))
        self.set_std_data(out, deterministic=True)

    def _draw(self, deterministic: bool = True) -> None:
        probas = self._get_parameter_value("probabilities")
        random = False if deterministic or self._deterministic else self.random_state
        self._index = int(discretization.softmax_discretization(probas, probas.size, random=random)[0])

    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        super().set_std_data(data, deterministic=deterministic)
        self._draw(deterministic=deterministic)
