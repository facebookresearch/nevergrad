# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict, Any
import numpy as np


class BaseFunction(abc.ABC):
    """Functions must inherit from this class for benchmarking purpose

    In child functions, implement:
    - "oracle_call": provides the output of your function (BaseFunction.__call__ will use it and add noise if noise_level > 0)
    - "get_description": add your own describers if your function takes different parameters.

    See ArtificialFunction for an example.
    """

    def __init__(self, dimension: int, noise_level: float = 0.) -> None:
        assert noise_level >= 0, "Noise level must be greater or equal to 0"
        assert dimension > 0
        assert isinstance(dimension, int)
        self._dimension = dimension
        self._noise_level = noise_level
        self._descriptors: Dict[str, Any] = {}
        self.add_descriptors(dimension=dimension, noise_level=noise_level, function_class=self.__class__.__name__)

    def __call__(self, x: np.ndarray) -> float:
        """Returns the output of the function,
        after adding noise if noise_level > 0
        Only overload this function if you want to change the noise pattern
        """
        noise_level = self._noise_level
        fx = self.oracle_call(x)
        if noise_level:
            fx += noise_level * np.random.normal(0, 1) * (self.oracle_call(x + np.random.normal(0, 1, self.dimension)) - fx)
        return fx

    def __repr__(self) -> str:
        """Shows the function name and its summary
        """
        params = [f"{x}={repr(y)}" for x, y in sorted(self.get_description().items())]
        return "Instance of {}({})".format(self.__class__.__name__, ", ".join(params))

    def __eq__(self, other: Any) -> bool:
        """Check that two instances where initialized with same settings.
        This is not meant to be used to check if functions are exactly equal (initialization may hold some randomness)
        This is only useful for unit testing.
        (may need to be overloaded to make faster if tests are getting slow)
        """
        if other.__class__ != self.__class__:
            return False
        return bool(self.get_description() == other.get_description())

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

    def add_descriptors(self, **kwargs: Any) -> None:
        """Add custom descriptors for the function.
        They will be returned with get_description
        """
        self._descriptors.update(kwargs)

    def get_description(self) -> Dict[str, Any]:
        """Describes the parameterization of the function

        Returns
        -------
        dict:
            a dict of parameters of the function. This basic implementation provides function_class,
            noise_level and dimension

        Note
        ----
        Append more describers to child classes. See ArtificialFunction for an example.
        """
        return dict(self._descriptors)
