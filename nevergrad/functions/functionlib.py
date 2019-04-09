# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from typing import List, Tuple, Any, Dict, Callable
import numpy as np
from . import utils
from . import corefuncs
from .. import instrumentation as inst
from ..common import tools
from ..common.typetools import ArrayLike


class ArtificialVariable(inst.var.utils.Variable[np.ndarray]):
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    # TODO: refactor, this is not more used for instrumentation, so using the
    # Variable framework is not necessary

    def __init__(self, dimension: int, num_blocks: int, block_dimension: int,
                 translation_factor: float, rotation: bool, hashing: bool, only_index_transform: bool) -> None:
        self._dimension = dimension
        self._transforms: List[utils.Transform] = []
        self.rotation = rotation
        self.translation_factor = translation_factor
        self.num_blocks = num_blocks
        self.block_dimension = block_dimension
        self.only_index_transform = only_index_transform
        self.hashing = hashing

    @property
    def dimension(self) -> int:
        return self._dimension if not self.hashing else 1

    def _initialize(self) -> None:
        """Delayed initialization of the transforms to avoid slowing down the instance creation
        (makes unit testing much faster).
        This functions creates the random transform used upon each block (translation + optional rotation).
        """
        # use random indices for blocks
        indices = np.random.choice(self._dimension, self.block_dimension * self.num_blocks, replace=False).tolist()
        indices.sort()  # keep the indices sorted sorted so that blocks do not overlap
        for transform_inds in tools.grouper(indices, n=self.block_dimension):
            self._transforms.append(utils.Transform(transform_inds, translation_factor=self.translation_factor, rotation=self.rotation))

    def process(self, data: ArrayLike, deterministic: bool = True) -> np.ndarray:  # pylint: disable=unused-argument
        if not self._transforms:
            self._initialize()
        if self.hashing:
            state = np.random.get_state()
            y = data[0]  # should be a string... or something...
            np.random.seed(int(int(hashlib.md5(y.encode()).hexdigest(), 16) % 500000))  # type: ignore
            data = np.random.normal(0., 1., len(y))  # type: ignore
            np.random.set_state(state)
        data = np.array(data, copy=False)
        output = []
        for transform in self._transforms:
            output.append(data[transform.indices] if self.only_index_transform else transform(data))
        return np.array(output)

    def _short_repr(self) -> str:
        return "Photonics"


class ArtificialFunction(inst.InstrumentedFunction, utils.PostponedObject, utils.NoisyBenchmarkFunction):
    """Artificial function object. This allows the creation of functions with different
    dimension and structure to be used for benchmarking in many different settings.

    Parameters
    ----------
    name: str
        name of the underlying function to use (like "sphere" for instance). If a wrong
        name is provided, an error is raised with all existing names.
    block_dimension: int
        the dimension on which the underlying function will be applied.
    num_blocks: int
        the number of blocks of size "block_dimension" on which the underlying function
        will be applied. The number of useful dimension is therefore num_blocks * core_dimension
    useless_variables: int
        the number of additional variables which have no impact on the core function.
        The full dimension of the function is therefore useless_variables + num_blocks * core_dimension
    noise_level: float
        noise level for the additive noise: noise_level * N(0, 1, size=1) * [f(x + N(0, 1, size=dim)) - f(x)]
    noise_dissymmetry: bool
        True if we dissymmetrize the model of noise
    rotation: bool
        whether the block space should be rotated (random rotation)
    hashing: bool
        whether the input data should be hashed
    aggregator: str
        how to aggregate the multiple block outputs

    Example
    -------
    >>> func = ArtificialFunction("sphere", 5, noise_level=.1)
    >>> x = [1, 2, 1, 0, .5]
    >>> func(x)  # returns a float
    >>> func(x)  # returns a different float since the function is noisy
    >>> func.oracle_call(x)   # returns a float
    >>> func.oracle_call(x)   # returns the same float (no noise for oracles + sphere function is deterministic)
    >>> func2 = ArtificialFunction("sphere", 5, noise_level=.1)
    >>> func2.oracle_call(x)   # returns a different float than before, because a random translation is applied

    Note
    ----
    - The full dimension of the function is available through the dimension attribute.
      Its value is useless_variables + num_blocks * block_dimension
    - The blocks are chosen with random sorted indices (blocks do not overlap)
    - A random translation is always applied to the function at initialization, so that
      instantiating twice the functions will give 2 different functions (unless you use
      seeding)
    - the noise formula is: noise_level * N(0, 1) * (f(x + N(0, 1)) - f(x))
    """

    def __init__(self, name: str, block_dimension: int, num_blocks: int = 1,  # pylint: disable=too-many-arguments
                 useless_variables: int = 0, noise_level: float = 0, noise_dissymmetry: bool = False,
                 rotation: bool = False, translation_factor: float = 1., hashing: bool = False,
                 aggregator: str = "max") -> None:
        # pylint: disable=too-many-locals
        self.name = name
        self._parameters = {x: y for x, y in locals().items() if x not in ["__class__", "self"]}
        # basic checks
        assert noise_level >= 0, "Noise level must be greater or equal to 0"
        if not all(isinstance(x, bool) for x in [noise_dissymmetry, hashing, rotation]):
            raise TypeError("hashing and rotation should be bools")
        for param, mini in [("block_dimension", 1), ("num_blocks", 1), ("useless_variables", 0)]:
            value = self._parameters[param]
            if not isinstance(value, int):
                raise TypeError(f'"{param}" must be an int')
            if value < mini:
                raise ValueError(f'"{param}" must be greater or equal to {mini}')
        if not isinstance(translation_factor, (float, int)):
            raise TypeError(f"Got non-float value {translation_factor}")
        if name not in corefuncs.registry:
            available = ", ".join(self.list_sorted_function_names())
            raise ValueError(f'Unknown core function "{name}". Available names are:\n-----\n{available}')
        # record necessary info and prepare transforms
        self._dimension = block_dimension * num_blocks + useless_variables
        self._func = corefuncs.registry[name]
        # special case
        info = corefuncs.registry.get_info(self._parameters["name"])
        only_index_transform = info.get("no_transfrom", False)
        # variable
        self.transform_var = ArtificialVariable(dimension=self._dimension, num_blocks=num_blocks, block_dimension=block_dimension,
                                                translation_factor=translation_factor, rotation=rotation, hashing=hashing,
                                                only_index_transform=only_index_transform)
        super().__init__(self.noisy_function, inst.var.Array(1 if hashing else self._dimension))
        self.instrumentation = self.instrumentation.with_name("")
        self._aggregator = {"max": np.max, "mean": np.mean, "sum": np.sum}[aggregator]
        info = corefuncs.registry.get_info(self._parameters["name"])
        # add descriptors
        self._descriptors.update(**self._parameters, useful_dimensions=block_dimension * num_blocks,
                                 discrete=any(x in name for x in ["onemax", "leadingones", "jump"]))
        # transforms are initialized at runtime to avoid slow init

    @property
    def dimension(self) -> int:
        return self._dimension  # bypass the instrumentation one (because of the "hashing" case)  # TODO: remove

    @staticmethod
    def list_sorted_function_names() -> List[str]:
        """Returns a sorted list of function names that can be used for the blocks
        """
        return sorted(corefuncs.registry)

    def _transform(self, x: ArrayLike) -> np.ndarray:
        data = self.transform_var.process(x)
        return np.array(data)

    def function_from_transform(self, x: np.ndarray) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        results = []
        for block in x:
            results.append(self._func(block))
        return float(self._aggregator(results))

    def noisefree_function(self, *args: Any, **kwargs: Any) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        assert len(args) == 1 and not kwargs
        data = self._transform(args[0])
        return self.function_from_transform(data)

    def noisy_function(self, x: ArrayLike) -> float:
        return _noisy_call(x=np.array(x, copy=False), transf=self._transform, func=self.function_from_transform,
                           noise_level=self._parameters["noise_level"], noise_dissymmetry=self._parameters["noise_dissymmetry"])

    def duplicate(self) -> "ArtificialFunction":
        """Create an equivalent instance, initialized with the same settings
        """
        return self.__class__(**self._parameters)

    def get_postponing_delay(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], value: float) -> float:
        """Delay before returning results in steady state mode benchmarks (fake execution time)
        """
        assert not kwargs
        assert len(args) == 1
        if isinstance(self._func, utils.PostponedObject):
            data = self._transform(args[0])
            total = 0.
            for block in data:
                total += self._func.get_postponing_delay((block,), {}, value)
            return total
        return 1.


def _noisy_call(x: np.ndarray, transf: Callable[[np.ndarray], np.ndarray], func: Callable[[np.ndarray], float],
                noise_level: float, noise_dissymmetry: bool) -> float:  # pylint: disable=unused-argument
    x_transf = transf(x)
    fx = func(x_transf)
    noise = 0
    if noise_level:
        if not noise_dissymmetry or x_transf.ravel()[0] <= 0:
            side_point = transf(x + np.random.normal(0, 1, size=len(x)))
            if noise_dissymmetry:
                noise_level *= (1. + x_transf.ravel()[0]*100.)
            noise = noise_level * np.random.normal(0, 1) * (func(side_point) - fx)
    return fx + noise
