import warnings
import typing as tp
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization.core import Descriptors
from nevergrad.common.typetools import ArrayLike
from .core import Variable, ArgsKwargs
# pylint: disable=no-value-for-parameter


VW = tp.TypeVar("VW", bound="VariableWrapper")


class VariableWrapper(p.Instrumentation):
    """Wrap a Variable to give it the Parameter interface
    """

    def __init__(self, variable: tp.Union[Variable, "VariableWrapper"]) -> None:
        super().__init__()
        self._variable: Variable = variable._variable if isinstance(variable, VariableWrapper) else variable
        self._data: np.ndarray = np.zeros((self._variable.dimension,))
        self._value: tp.Optional[ArgsKwargs] = None

    def cheap_constraint_check(self, *args: tp.Any, **kwargs: tp.Any) -> bool:
        return self._variable._constraint_checker(*args, **kwargs)

    def set_cheap_constraint_checker(self, func: tp.Callable[..., bool]) -> None:
        self._variable._constraint_checker = func

    def _get_name(self) -> str:
        return self._variable.name

    @property  # type: ignore
    def value(self) -> ArgsKwargs:  # type: ignore
        if self._value is None:
            self._value = self.data_to_arguments(self._data)
        return self._value

    @value.setter
    def value(self, value: ArgsKwargs) -> None:
        self._value = value
        self._data = self.arguments_to_data(*value[0], **value[1])

    @property
    def args(self) -> tp.Tuple[tp.Any, ...]:
        return self.value[0]

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        return self.value[1]

    def _internal_get_std_data(self: VW, instance: VW) -> np.ndarray:
        return instance._data

    def _internal_set_std_data(self: VW, data: np.ndarray, instance: VW, deterministic: bool = False) -> VW:
        instance._data = data
        self._value = instance.data_to_arguments(self._data, deterministic=deterministic)
        return instance

    def _internal_spawn_child(self: VW) -> VW:
        inst = self.__class__(self._variable)
        inst._data = self._data
        inst._value = self.data_to_arguments(self._data)
        return inst

    @property
    def random_state(self) -> np.random.RandomState:
        return self._variable.random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        self._variable.random_state = random_state

    @property
    def noisy(self) -> bool:
        return self._variable.noisy

    @property
    def continuous(self) -> bool:
        return self._variable.continuous

    @property
    def descriptors(self) -> Descriptors:
        return Descriptors(continuous=self.continuous, deterministic=not self.noisy)

    def arguments_to_data(self, *args: tp.Any, **kwargs: tp.Any) -> np.ndarray:
        """Converts args and kwargs into data in np.ndarray format
        """
        return self._variable.arguments_to_data(*args, **kwargs)

    def data_to_arguments(self, data: ArrayLike, deterministic: bool = False) -> ArgsKwargs:
        """Converts data to arguments
        Parameters
        ----------
        data: ArrayLike (list/tuple of floats, np.ndarray)
            the data in the optimization space
        deterministic: bool
            whether the conversion should be deterministic (some variables can be stochastic, if deterministic=True
            the most likely output will be used)
        Returns
        -------
        args: Tuple[Any]
            the positional arguments corresponding to the instance initialization positional arguments
        kwargs: Dict[str, tp.Any]
            the keyword arguments corresponding to the instance initialization keyword arguments
        """
        # trigger random_state creation (may require to be propagated to sub-variables
        assert self.random_state is not None
        return self._variable.data_to_arguments(data, deterministic=deterministic)

    def get_summary(self, data: ArrayLike) -> str:  # pylint: disable=unused-argument
        warnings.warn("get_summary will disappear since new parameters are easier to analyze directly")
        output = self.data_to_arguments(np.array(data, copy=False), deterministic=True)
        return f"Value {output[0][0]}, from data: {data}"

    def mutate(self) -> None:
        raise p.NotSupportedError("Please port your code to new parametrization")

    def recombine(self: VW, *others: VW) -> None:  # type: ignore
        raise p.NotSupportedError("Please port your code to new parametrization")
