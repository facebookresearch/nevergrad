# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict, Any, Callable, Optional
import numpy as np


class BaseFunction(abc.ABC):
    """Functions must inherit from this class for benchmarking purpose
    In child functions, implement "oracle_call". This method should provide the output of your function
    (BaseFunction.__call__ will use it and add noise if noise_level > 0)
    Also, update "_descriptors" dict attribute so that function parametrization is recorded during benchmark.
    See ArtificialFunction for an example.

    Parameters
    ----------
    dimension: int
        dimension of the input space.
    noise_level: float
        level of the noise to add
    transform: optional str
        name of a registered transform to be applied to the input data.

    Notes
    -----
    - the noise formula is: noise_level * N(0, 1) * (f(x + N(0, 1)) - f(x))
    - transforms must be registered through the "register_transform" class method before instanciation.
    """

    _TRANSFORMS: Dict[str, Callable[[Any, np.ndarray], np.ndarray]] = {}  # Any should be the current class (but typing would get messy)

    def __init__(self, dimension: int, noise_level: float = 0., transform: Optional[str] = None) -> None:
        assert noise_level >= 0, "Noise level must be greater or equal to 0"
        assert dimension > 0
        assert isinstance(dimension, int)
        self._dimension = dimension
        self._transform = transform
        self._noise_level = noise_level
        self._descriptors: Dict[str, Any] = {}
        self._descriptors.update(dimension=dimension, noise_level=noise_level, function_class=self.__class__.__name__, transform=transform)
        if transform is not None and transform not in self._TRANSFORMS:
            raise ValueError(f'Unknown transform "{self._transform}", available are:\n{list(self._TRANSFORMS.keys())}\n'
                             f'(you must register new ones with "{self.__class__.__name__}.register_transform" before instanciation)')

    @classmethod
    def register_transform(cls, name: str, func: Callable[["BaseFunction", np.ndarray], np.ndarray]) -> None:
        """Register a transform for use in call.

        Parameters
        ----------
        name: str
            name of the transform (this will be used as descriptor)
        func: callable
            A callable with the function as first input and point as second input, returning the transformed point.
        """
        cls._TRANSFORMS[name] = func

    @property
    def descriptors(self) -> Dict[str, Any]:
        """Description of the function parametrizaion, as a dict. This base class implementation provides function_class,
            noise_level, transform and dimension
        """
        return dict(self._descriptors)  # Avoid external modification

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform the input to another function specific domain.
        """
        if self._transform is not None:
            x = self._TRANSFORMS[self._transform](self, x)
        return x

    def __call__(self, x: np.ndarray) -> float:
        """Returns the output of the function,
        after adding noise if noise_level > 0 and transforming the data
        Only overload this function if you want to change the noise pattern
        """
        x = self.transform(x)
        noise_level = self._noise_level
        fx = self.oracle_call(x)
        if noise_level:
            fx += noise_level * np.random.normal(0, 1) * (self.oracle_call(x + np.random.normal(0, 1, self.dimension)) - fx)
        return fx

    def __repr__(self) -> str:
        """Shows the function name and its summary
        """
        params = [f"{x}={repr(y)}" for x, y in sorted(self._descriptors.items())]
        return "Instance of {}({})".format(self.__class__.__name__, ", ".join(params))

    def __eq__(self, other: Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
        if other.__class__ != self.__class__:
            return False
        return bool(self._descriptors == other._descriptors)

    @property
    def dimension(self) -> int:
        """Dimension of the input space
        """
        return self._dimension

    @abc.abstractmethod
    def oracle_call(self, x: np.ndarray) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.

        Note
        ----
        "oracle_call" is not necessarily deterministic
        """
        raise NotImplementedError
