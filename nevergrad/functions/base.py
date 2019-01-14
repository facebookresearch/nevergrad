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
    (BaseFunction.__call__ will use it, and call the _add_noise method if you implemented it)
    Also, update "_descriptors" dict attribute so that function parameterization is recorded during benchmark.
    See ArtificialFunction for an example.

    Parameters
    ----------
    dimension: int
        dimension of the input space.
    transform: optional str
        name of a registered transform to be applied to the input data.

    Notes
    -----
    - transforms must be registered through the "register_transform" class method before instantiation.
    """

    _TRANSFORMS: Dict[str, Callable[[Any, np.ndarray], np.ndarray]] = {}  # Any should be the current class (but typing would get messy)

    def __init__(self, dimension: int, transform: Optional[str] = None) -> None:
        assert dimension > 0
        assert isinstance(dimension, int)
        self._dimension = dimension
        self._transform = transform
        self._descriptors: Dict[str, Any] = {}
        self._descriptors.update(dimension=dimension, function_class=self.__class__.__name__, transform=transform)
        if transform is not None and transform not in self._TRANSFORMS:
            raise ValueError(f'Unknown transform "{self._transform}", available are:\n{list(self._TRANSFORMS.keys())}\n'
                             f'(you must register new ones with "{self.__class__.__name__}.register_transform" before instantiation)')

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
        """Description of the function parameterization, as a dict. This base class implementation provides function_class,
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
        after transforming the data and adding noise through _add_noise
        (by default, _add_noise does not add any noise).
        It is preferable to avoid overloading this function in order to avoid issues
        with transformations and noise. Override _add_noise and oracle_call instead.
        """
        x_transf = self.transform(x)
        fx = self.oracle_call(x_transf)
        noisy_fx = self._add_noise(x, x_transf, fx)
        return noisy_fx

    def _add_noise(self, x_input: np.ndarray, x_transf: np.ndarray, fx: float) -> float:  # pylint: disable=unused-argument
        """Adds noise to the output of the function
        This is useful for artificial functions only.

        Parameters
        ----------
        x_input: np.ndarray
            Input point, before transformation
        x_transf: np.nparray
            Input point, after transformation
        fx: float
            Output before noise, returned by oracle_call
        """
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
        Under the hood, __call__ delegates to oracle_call + applies the transform and add some noise if need be.

        Parameter
        ---------
        x: np.ndarray
            The input data *before* transformation.

        Notes
        -----
        - "oracle_call" is not necessarily deterministic
        - the transform is applied *before* this function, do not apply it here.

        """
        raise NotImplementedError


class ArtificiallyNoisyBaseFunction(BaseFunction):  # pylint: disable=abstract-method
    """Functions must inherit from this class for benchmarking purpose
    In child functions, implement "oracle_call". This method should provide the output of your function
    (BaseFunction.__call__ will use it and add noise if noise_level > 0)
    Also, update "_descriptors" dict attribute so that function parameterization is recorded during benchmark.
    See ArtificialFunction for an example.

    Parameters
    ----------
    dimension: int
        dimension of the input space.
    noise_level: float
        level of the noise to add
    noise_dissymmetry: bool
        True if we dissymetrize the noise model
    transform: optional str
        name of a registered transform to be applied to the input data.

    Notes
    -----
    - the noise formula is: noise_level * N(0, 1) * (f(x + N(0, 1)) - f(x))
    - transforms must be registered through the "register_transform" class method before instantiation.
    """

    def __init__(self, dimension: int, noise_level: float = 0., noise_dissymmetry: bool = False, transform: Optional[str] = None) -> None:
        super().__init__(dimension, transform=transform)
        assert noise_level >= 0, "Noise level must be greater or equal to 0"
        self._noise_level = noise_level
        self._noise_dissymmetry = noise_dissymmetry
        self._descriptors.update(noise_level=noise_level, noise_dissymmetry=noise_dissymmetry)

    def _add_noise(self, x_input: np.ndarray, x_transf: np.ndarray, fx: float) -> float:  # pylint: disable=unused-argument
        noise = 0
        noise_level = self._noise_level
        if noise_level:
            if not self._noise_dissymmetry or x_input[0] <= 0:
                side_point = self.transform(x_input + np.random.normal(0, 1, self.dimension))
                noise = noise_level * np.random.normal(0, 1) * (self.oracle_call(side_point) - fx)
        return fx + noise
