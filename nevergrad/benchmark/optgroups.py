# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Groups of optimizers for use in benchmarks
"""
import typing as tp
import nevergrad as ng
from nevergrad.common.decorators import Registry
from nevergrad.optimization import base as obase

Optim = tp.Union[obase.ConfiguredOptimizer, str]
registry = Registry[tp.Callable[[], tp.Iterable[Optim]]]()


def get_optimizers(*names: str) -> tp.List[Optim]:
    optims: tp.List[Optim] = []
    for name in names:
        for optim in registry[name]():
            if optim not in optims:  # avoid duplicates
                optims.append(optim)
    return optims


@registry.register
def large() -> tp.List[Optim]:
    return ["NGO", "Shiva", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE",
            "OnePlusOne", "SQP", "Cobyla", "Powell",
            "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE",
            "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS", "CM",
            "MultiCMA", "TripleCMA", "MultiScaleCMA", "RSQP", "RCobyla", "RPowell", "SQPCMA"]


@registry.register
def cma() -> tp.List[Optim]:
    return ["DiagonalCMA", "CMA"]


@registry.register
def competence_map() -> tp.List[Optim]:
    return ["NGO", "Shiva"]


@registry.register
def competitive() -> tp.List[Optim]:
    """A set of competitive algorithms
    """
    return get_optimizers("cma", "competence_map") + ["NaiveTBPSA", "PSO", "DE", "LhsDE", "RandomSearch", "OnePlusOne", "TwoPointsDE"]


@registry.register
def all_bo() -> tp.List[Optim]:
    return sorted(x for x in ng.optimizers.registry if "BO" in x)


@registry.register
def spsa() -> tp.List[Optim]:
    # return sorted(x for x, y in ng.optimizers.registry.items() if (any(e in x for e in "TBPSA SPSA".split()) and "iscr" not in x))
    return ["NaiveTBPSA", "SPSA", "TBPSA"]
