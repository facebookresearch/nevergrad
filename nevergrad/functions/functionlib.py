# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import itertools
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools
import nevergrad.common.typing as tp
from .base import ExperimentFunction
from .multiobjective import MultiobjectiveFunction
from .pbt import PBT as PBT  # pylint: disable=unused-import
from . import utils
from . import corefuncs


class ArtificialVariable:
    # pylint: disable=too-many-instance-attributes,too-many-arguments
    # TODO: refactor, this is not more used for parametrization, so using the
    # Variable framework is not necessary

    def __init__(self, dimension: int, num_blocks: int, block_dimension: int,
                 translation_factor: float, rotation: bool, hashing: bool, only_index_transform: bool) -> None:
        self._dimension = dimension
        self._transforms: tp.List[utils.Transform] = []
        self.rotation = rotation
        self.translation_factor = translation_factor
        self.num_blocks = num_blocks
        self.block_dimension = block_dimension
        self.only_index_transform = only_index_transform
        self.hashing = hashing
        self.dimension = self._dimension if not self.hashing else 1  # external dim?

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

    def process(self, data: tp.ArrayLike, deterministic: bool = True) -> np.ndarray:  # pylint: disable=unused-argument
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


class ArtificialFunction(ExperimentFunction):
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
        whether the input data should be hashed. In this case, the function expects an array of size 1 with
        string as element.
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
        only_index_transform = info.get("no_transform", False)
        # variable
        self.transform_var = ArtificialVariable(dimension=self._dimension, num_blocks=num_blocks, block_dimension=block_dimension,
                                                translation_factor=translation_factor, rotation=rotation, hashing=hashing,
                                                only_index_transform=only_index_transform)
        parametrization = p.Array(shape=(1,) if hashing else (self._dimension,)).set_name("")
        if noise_level > 0:
            parametrization.descriptors.deterministic_function = False
        super().__init__(self.noisy_function, parametrization)
        self.register_initialization(**self._parameters)
        self._aggregator = {"max": np.max, "mean": np.mean, "sum": np.sum}[aggregator]
        info = corefuncs.registry.get_info(self._parameters["name"])
        # add descriptors
        self._descriptors.update(**self._parameters, useful_dimensions=block_dimension * num_blocks,
                                 discrete=any(x in name for x in ["onemax", "leadingones", "jump"]))
        # transforms are initialized at runtime to avoid slow init
        if hasattr(self._func, "get_postponing_delay"):
            raise RuntimeError('"get_posponing_delay" has been replaced by "compute_pseudotime" and has been  aggressively deprecated')

    @property
    def dimension(self) -> int:
        return self._dimension  # bypass the parametrization one (because of the "hashing" case)  # TODO: remove

    @staticmethod
    def list_sorted_function_names() -> tp.List[str]:
        """Returns a sorted list of function names that can be used for the blocks
        """
        return sorted(corefuncs.registry)

    def _transform(self, x: tp.ArrayLike) -> np.ndarray:
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

    def evaluation_function(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        assert len(args) == 1 and not kwargs
        data = self._transform(args[0])
        return self.function_from_transform(data)

    def noisy_function(self, x: tp.ArrayLike) -> float:
        return _noisy_call(x=np.array(x, copy=False), transf=self._transform, func=self.function_from_transform,
                           noise_level=self._parameters["noise_level"], noise_dissymmetry=self._parameters["noise_dissymmetry"])

    def compute_pseudotime(self, input_parameter: tp.Any, value: float) -> float:
        """Delay before returning results in steady state mode benchmarks (fake execution time)
        """
        args, kwargs = input_parameter
        assert not kwargs
        assert len(args) == 1
        if hasattr(self._func, "compute_pseudotime"):
            data = self._transform(args[0])
            total = 0.
            for block in data:
                total += self._func.compute_pseudotime(((block,), {}), value)  # type: ignore
            return total
        return 1.


def _noisy_call(x: np.ndarray, transf: tp.Callable[[np.ndarray], np.ndarray], func: tp.Callable[[np.ndarray], float],
                noise_level: float, noise_dissymmetry: bool) -> float:  # pylint: disable=unused-argument
    x_transf = transf(x)
    fx = func(x_transf)
    noise = 0
    if noise_level:
        if not noise_dissymmetry or x_transf.ravel()[0] <= 0:
            side_point = transf(x + np.random.normal(0, 1, size=len(x)))
            if noise_dissymmetry:
                noise_level *= (1. + x_transf.ravel()[0] * 100.)
            noise = noise_level * np.random.normal(0, 1) * (func(side_point) - fx)
    return fx + noise


class FarOptimumFunction(ExperimentFunction):
    """Very simple 2D norm-1 function with optimal value at (x_optimum, 100)
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            independent_sigma: bool = True,
            mutable_sigma: bool = True,
            multiobjective: bool = False,
            recombination: str = "crossover",
            optimum: tp.Tuple[int, int] = (80, 100)
    ) -> None:
        assert recombination in ("crossover", "average")
        self._optimum = np.array(optimum, dtype=float)
        parametrization = p.Array(shape=(2,), mutable_sigma=mutable_sigma)
        init = np.array([1.0, 1.0] if independent_sigma else [1.0], dtype=float)
        sigma = (
            p.Array(init=init).set_mutation(exponent=2.0)
            if mutable_sigma else p.Constant(init)
        )
        parametrization.set_mutation(sigma=sigma)
        parametrization.set_recombination("average" if recombination == "average" else p.mutation.Crossover())
        self._multiobjective = MultiobjectiveFunction(self._multifunc, 2 * self._optimum)
        super().__init__(self._multiobjective if multiobjective else self._monofunc, parametrization.set_name(""))  # type: ignore
        descr = dict(independent_sigma=independent_sigma, mutable_sigma=mutable_sigma,
                     multiobjective=multiobjective, optimum=optimum, recombination=recombination)
        self._descriptors.update(descr)
        self.register_initialization(**descr)

    def _multifunc(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x - self._optimum)  # type: ignore

    def _monofunc(self, x: np.ndarray) -> float:
        return float(np.sum(self._multifunc(x)))

    def evaluation_function(self, *args: tp.Any, **kwargs: tp.Any) -> float:
        return self._monofunc(args[0])

    @classmethod
    def itercases(cls) -> tp.Iterator["FarOptimumFunction"]:
        options = dict(independent_sigma=[True, False],
                       mutable_sigma=[True, False],
                       multiobjective=[True, False],
                       recombination=["average", "crossover"],
                       optimum=[(.8, 1), (80, 100), (.8, 100)]
                       )
        keys = sorted(options)
        select = itertools.product(*(options[k] for k in keys))  # type: ignore
        cases = (dict(zip(keys, s)) for s in select)
        return (cls(**c) for c in cases)
