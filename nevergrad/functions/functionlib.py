# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from typing import List
import numpy as np
from . import utils
from . import corefuncs
from .base import BaseFunction
from ..common import tools


class ArtificialFunction(BaseFunction):
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
    """

    def __init__(self, name: str, block_dimension: int, num_blocks: int = 1,  # pylint: disable=too-many-arguments
                 useless_variables: int = 0, noise_level: float = 0, rotation: bool = False,
                 translation_factor: int = 1, hashing: bool = False, aggregator: str = "max") -> None:
        # pylint: disable=too-many-locals
        self._parameters = {x: y for x, y in locals().items() if x not in ["__class__", "self"]}
        # basic checks
        assert all(isinstance(x, bool) for x in [hashing, rotation])
        for param, mini in [("block_dimension", 1), ("num_blocks", 1), ("useless_variables", 0)]:
            value = locals()[param]
            assert isinstance(value, int), f'"{param}" must be an int'
            assert value >= mini, f'"{param}" must be greater or equal to {mini}'
        for param in ["hashing", "rotation"]:
            assert isinstance(locals()[param], bool)
        assert isinstance(translation_factor, (float, int)), f"Got non-float value {translation_factor}"
        if name not in corefuncs.registry:
            available = ", ".join(self.list_sorted_function_names())
            raise ValueError(f'Unknown core function "{name}". Avaible names are:\n-----\n{available}')
        # record necessary info and prepare transforms
        dimension = block_dimension * num_blocks + useless_variables
        super().__init__(dimension, noise_level)
        self._aggregator = {"max": max, "mean": np.mean, "sum": sum}[aggregator]
        self._transforms: List[utils.Transform] = []
        # special case
        info = corefuncs.registry.get_info(self._parameters["name"])
        self._only_index_transform = info.get("no_transfrom", False)
        # add descriptors
        self._descriptors.update(**self._parameters, useful_dimensions=block_dimension * num_blocks,
                                 discrete=any(x in name for x in ["onemax", "leadingones", "jump"]))
        # transforms are initialized at runtime to avoid slow init

    @staticmethod
    def list_sorted_function_names() -> List[str]:
        """Returns a sorted list of function names that can be used for the blocks
        """
        return sorted(corefuncs.registry)

    def initialize(self) -> None:
        """Delayed initializations of the transforms to avoid slowing down the instance creation
        (makes unit testing much faster).
        This functions creates the random transform used upon each block (translation + optional rotation).
        """
        # use random indices for blocks
        indices = np.random.choice(self.dimension, self.dimension - self._parameters["useless_variables"], replace=False).tolist()
        indices.sort()  # keep the indices sorted sorted so that blocks do not overlap
        for transform_inds in tools.grouper(indices, n=self._parameters["block_dimension"]):
            self._transforms.append(utils.Transform(transform_inds, **{x: self._parameters[x] for x in ["translation_factor", "rotation"]}))

    def oracle_call(self, x: np.ndarray) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        if not self._transforms:
            self.initialize()
        if self._parameters["hashing"]:
            state = np.random.get_state()
            np.random.seed(int(int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 500000))
            x = np.random.normal(0., 1., len(x))
            np.random.set_state(state)
        x = np.asarray(x)
        results = []
        for transform in self._transforms:
            translated_x = transform(x)
            if self._only_index_transform:
                translated_x = x[transform.indices]  # only subsampling in this case
            results.append(corefuncs.registry[self._parameters["name"]](translated_x))
        return float(self._aggregator(results))

    def duplicate(self) -> "ArtificialFunction":
        """Create an equivalent instance, initialized with the same settings
        """
        return self.__class__(**self._parameters)  # type: ignore
