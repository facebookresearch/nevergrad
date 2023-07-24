# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is based on the specifications of:
# X. Li, K. Tang, M. Omidvar, Z. Yang and K. Qin,
# "Benchmark Functions for the CEC’2013 Special Session and Competition on Large Scale Global Optimization",
# Technical Report, Evolutionary Computation and Machine Learning Group, RMIT University, Australia, 2013.
# and the corresponding source code.

from typing import List, Callable, Dict, Type, Optional
from pathlib import Path
import re
import numpy as np
from nevergrad.common.decorators import Registry
from nevergrad.functions import ExperimentFunction
import nevergrad as ng
from . import _core


registry = Registry[Type["LsgoFunction"]]()


def read_data(name: str) -> Dict[str, np.ndarray]:
    filepaths = (Path(__file__).parent / "cdatafiles").glob(name + "-*.txt")
    all_data: Dict[str, np.ndarray] = {}
    pattern = re.compile(name + "-(?P<tag>.+?).txt")
    for filepath in filepaths:
        match = pattern.search(filepath.name)
        assert match is not None
        tag = match.group("tag")
        all_data[tag] = np.loadtxt(str(filepath), delimiter=",", dtype=int if tag in ["s", "p"] else float)
    if "p" in all_data:
        all_data["p"] -= 1  # indexing starts at 0 in Python
    return all_data


class FunctionChunk:
    """Computes a sequence of transforms and applies a loss on it.
    This can be left multiplied by a scalar to add a weight.
    """

    def __init__(
        self, transforms: List[Callable[[np.ndarray], np.ndarray]], loss: Callable[[np.ndarray], float]
    ) -> None:
        self.loss = loss
        self.transforms = transforms
        self._scalar = 1.0

    def _apply_transforms(self, x: np.ndarray) -> np.ndarray:
        if not self.transforms:
            return x
        y = self.transforms[0](x)
        for transf in self.transforms[1:]:
            y = transf(y)
        return y

    def __call__(self, x: np.ndarray) -> float:
        return self._scalar * self.loss(self._apply_transforms(x))

    def __rmul__(self, scalar: float) -> "FunctionChunk":
        self._scalar *= scalar
        return self

    def __repr__(self) -> str:
        return f"{self._scalar}*FunctionChunck({self.transforms}, {self.loss}"


class LsgoFunction:
    """Base function used in the LSGO testbed.
    It is in charge of computing the translation, and summing all function chunks.
    The "optimum" attribute holds the zero of the function. It is usually xopt, but is xopt + 1 in one case (F12),
    and unknown in one other (F14).
    """

    bounds = (-100, 100)
    tags = ["unimodal", "separable", "shifted", "irregular"]
    conditionning = -1.0

    def __init__(self, xopt: np.ndarray, functions: List[FunctionChunk]) -> None:
        self.xopt = xopt
        self.optimum: Optional[np.ndarray] = xopt  # default to xopt, but requires changing for rosenbroc
        self.functions = functions

    def __call__(self, x: np.ndarray) -> float:
        transformed = x - self.xopt
        return sum(f(transformed) for f in self.functions)

    @property
    def dimension(self) -> int:
        """Dimension of the function space"""
        return self.xopt.size

    def instrumented(self, transform: str = "bouncing") -> ExperimentFunction:
        """Returns an instrumented function, taking the bounds into account by composing it with
        a bounding transform. Instrumentated functions are necessary
        for nevergrad benchmarking

        Parameter
        ---------
        transform: str
            "bouncing", "arctan", "tanh" or "clipping"
        """
        param = ng.p.Array(shape=(self.dimension,)).set_bounds(
            self.bounds[0], self.bounds[1], method=transform
        )
        return ExperimentFunction(self, param.set_name(transform))


@registry.register
class ShiftedElliptic(LsgoFunction):

    number = 1
    bounds = (-100, 100)
    conditionning = 1e6
    tags = ["unimodal", "separable", "shifted", "irregular"]

    def __init__(self, xopt: np.ndarray) -> None:
        transforms: List[Callable[[np.ndarray], np.ndarray]] = [_core.irregularity]
        super().__init__(xopt, [FunctionChunk(transforms, _core.Elliptic(xopt.size))])


@registry.register
class ShiftedRastrigin(LsgoFunction):

    number = 2
    bounds = (-5, 5)
    conditionning = 10.0
    tags = ["multimodal", "separable", "shifted", "irregular"]

    def __init__(self, xopt: np.ndarray) -> None:
        transforms: List[Callable[[np.ndarray], np.ndarray]] = [
            _core.irregularity,
            _core.Asymmetry(0.2),
            _core.Illconditionning(10.0),
        ]
        super().__init__(xopt, [FunctionChunk(transforms, _core.rastrigin)])


@registry.register
class ShiftedAckley(LsgoFunction):

    number = 3
    bounds = (-32, 32)
    conditionning = 10.0
    tags = ["multimodal", "separable", "shifted", "irregular"]

    def __init__(self, xopt: np.ndarray) -> None:
        transforms: List[Callable[[np.ndarray], np.ndarray]] = [
            _core.irregularity,
            _core.Asymmetry(0.2),
            _core.Illconditionning(10.0),
        ]
        super().__init__(xopt, [FunctionChunk(transforms, _core.ackley)])


# %% PARTIALLY SEPARABLE 1 %% #


class _MultiPartFunction(LsgoFunction):
    """Base class for most multi-part function, overlapping or not."""

    number = -1
    bounds = (-100, 100)
    tags: List[str] = []
    conditionning = -1.0
    overlap = 0

    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]:
        raise NotImplementedError

    def _make_transforms(self, side_loss: bool) -> List[Callable[[np.ndarray], np.ndarray]]:
        raise NotImplementedError

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        xopt: np.ndarray,
        p: np.ndarray,
        s: np.ndarray,
        w: np.ndarray,
        R25: np.ndarray,
        R50: np.ndarray,
        R100: np.ndarray,
    ) -> None:
        remaining = 1000 - np.sum(s)
        assert remaining in [0, 700]
        assert xopt.size == p.size
        indexings = _core.Indexing.from_split(
            p, s.tolist() + ([remaining] if remaining else []), overlap=self.overlap
        )
        rotations = {num: _core.Rotation(rot) for num, rot in [(25, R25), (50, R50), (100, R100)]}
        assert w.size == len(indexings) - (remaining == 700)
        functions: List[FunctionChunk] = []
        transforms: List[Callable[[np.ndarray], np.ndarray]] = []
        for coeff, indexing in zip(w.tolist(), indexings):
            transforms = [indexing, rotations[indexing.outdim]] + self._make_transforms(side_loss=False)  # type: ignore
            functions.append(
                coeff * FunctionChunk(transforms, self._make_loss(indexing.outdim, side_loss=False))
            )
        if remaining:
            transforms = self._make_transforms(side_loss=True)
            loss = self._make_loss(indexings[-1].outdim, side_loss=True)
            functions.append(FunctionChunk([indexings[-1]] + transforms, loss))  # type: ignore
        super().__init__(xopt, functions)


@registry.register
class PartiallySeparableElliptic(_MultiPartFunction):

    number = 4
    bounds = (-100, 100)
    tags = ["unimodal", "partially separable", "shifted", "irregularities"]
    conditionning = 1e6

    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]:
        return _core.Elliptic(dimension)

    def _make_transforms(self, side_loss: bool) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [_core.irregularity]


@registry.register
class PartiallySeparableRastrigin(_MultiPartFunction):

    number = 5
    bounds = (-5, 5)
    tags = ["multimodal", "partially separable", "shifted", "irregularities"]
    conditionning = 10.0

    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]:
        return _core.rastrigin

    def _make_transforms(self, side_loss: bool) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [_core.irregularity, _core.Asymmetry(0.2), _core.Illconditionning(10.0)]


@registry.register
class PartiallySeparableAckley(_MultiPartFunction):

    number = 6
    bounds = (-32, 32)
    tags = ["multimodal", "partially separable", "shifted", "irregularities"]
    conditionning = 10.0

    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]:
        return _core.ackley

    def _make_transforms(self, side_loss: bool) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [_core.irregularity, _core.Asymmetry(0.2), _core.Illconditionning(10.0)]


@registry.register
class PartiallySeparableSchwefel(_MultiPartFunction):

    number = 7
    bounds = (-100, 100)
    tags = ["multimodal", "partially separable", "shifted", "irregularities"]
    conditionning = -1.0  # not provided

    def _make_loss(self, dimension: int, side_loss: bool) -> Callable[[np.ndarray], float]:
        return _core.sphere if side_loss else _core.schwefel_1_2

    def _make_transforms(self, side_loss: bool) -> List[Callable[[np.ndarray], np.ndarray]]:
        return (
            [] if side_loss else [_core.irregularity, _core.Asymmetry(0.2)]
        )  # this reproduces implementations, not paper


# %% PARTIALLY SEPARABLE 2 %% #


@registry.register
class PartiallySeparableElliptic2(PartiallySeparableElliptic):

    number = 8
    bounds = (-100, 100)
    tags = ["unimodal", "partially separable", "shifted", "irregularities"]
    conditionning = 1e6


@registry.register
class PartiallySeparableRastrigin2(PartiallySeparableRastrigin):

    number = 9
    bounds = (-5, 5)
    tags = ["multimodal", "partially separable", "shifted", "irregularities"]
    conditionning = 10.0


@registry.register
class PartiallySeparableAckley2(PartiallySeparableAckley):

    number = 10
    bounds = (-32, 32)
    tags = ["multimodal", "partially separable", "shifted", "irregularities"]
    conditionning = 10.0


@registry.register
class PartiallySeparableSchwefel2(PartiallySeparableSchwefel):

    number = 11
    bounds = (-100, 100)
    tags = ["unimodal", "partially separable", "shifted", "irregularities"]
    conditionning = -1.0  # not provided


# %% Overlapping functions %% #


@registry.register
class ShiftedRosenbrock(LsgoFunction):

    number = 12
    bounds = (-100, 100)
    conditionning = -1.0  # not provided
    tags = ["multimodal", "separable", "shifted", "irregular"]

    def __init__(self, xopt: np.ndarray) -> None:
        super().__init__(xopt, [FunctionChunk([], _core.rosenbrock)])
        self.optimum = xopt + 1  # xopt and optimum are different


@registry.register
class OverlappingSchwefel(PartiallySeparableSchwefel):

    number = 13
    bounds = (-100, 100)
    tags = ["unimodal", "non-separable", "overlapping", "shifted", "irregularities"]
    overlap = 5


@registry.register
class ConflictingSchwefel(LsgoFunction):

    number = 14
    bounds = (-100, 100)
    tags = ["unimodal", "non-separable", "conflicting", "shifted", "irregularities"]
    conditionning = -1.0  # not provided

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        xopt: np.ndarray,
        p: np.ndarray,
        s: np.ndarray,
        w: np.ndarray,
        R25: np.ndarray,
        R50: np.ndarray,
        R100: np.ndarray,
    ) -> None:
        assert xopt.size == 1000
        assert p.size == 905
        indexings = _core.Indexing.from_split(p, s.tolist(), overlap=5)
        xopt_indexings = _core.Indexing.from_split(np.arange(1000), s.tolist(), overlap=0)
        rotations = {num: _core.Rotation(rot) for num, rot in [(25, R25), (50, R50), (100, R100)]}
        assert w.size == len(indexings)
        functions: List[FunctionChunk] = []
        transforms: List[Callable[[np.ndarray], np.ndarray]] = []
        for coeff, indexing, xind in zip(w.tolist(), indexings, xopt_indexings):
            transforms = [
                indexing,
                _core.Translation(xopt[xind.indices]),
                rotations[indexing.outdim],
                _core.irregularity,
                _core.Asymmetry(0.2),
            ]
            functions.append(coeff * FunctionChunk(transforms, _core.schwefel_1_2))
        super().__init__(np.zeros((905,)), functions)
        self.optimum = None  # unknown


@registry.register
class ShiftedSchwefel(LsgoFunction):

    number = 15
    bounds = (-100, 100)
    tags = ["unimodal", "fully non-separable", "shifted", "irregular"]
    conditionning = -1.0  # not provided

    def __init__(self, xopt: np.ndarray) -> None:
        transforms: List[Callable[[np.ndarray], np.ndarray]] = [_core.irregularity, _core.Asymmetry(0.2)]
        super().__init__(xopt, [FunctionChunk(transforms, _core.schwefel_1_2)])


def make_function(number: int) -> LsgoFunction:
    """Creates one of the LSGO functions.

    Parameters
    ----------
    number: int
        the number of the function, from 1 to 15 (included)

    Returns
    -------
    LsgoFunction
        A function which acts exactly as the CPP implementation of LSGO (which may deviate from matlab or the
        actual paper). It has an attribute dimension for the optimization space dimension, and bounds for the upper
        and lower bounds of this space.
    """
    if not number:
        raise ValueError("Numbering of LSGO functions starts from 1")
    num_registry = {cls.number: cls for cls in registry.values()}  # type: ignore
    return num_registry[number](**read_data(f"F{number}"))  # type: ignore
