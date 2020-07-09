# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Groups of optimizers for use in benchmarks
"""
import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.common.decorators import Registry
from nevergrad.optimization import base as obase

Optim = tp.Union[obase.ConfiguredOptimizer, str]
registry: Registry[tp.Callable[[], tp.Iterable[Optim]]] = Registry()


def get_optimizers(*names: str, seed: tp.Optional[int] = None) -> tp.List[Optim]:
    """Returns an deterministically ordered list of optimizers, belonging to the
    provided groups. If a seed is provided, it is used to shuffle the list.

    Parameters
    ----------
    *names: str
        names of the groups to use. See nevergrad/benchmarks/optimizer_groups.txt
        (generated/updated when running pytest)
        for the list of groups and what they contain
    seed: optional int
        a seed to shuffle the list of optimizers
    """
    optims: tp.List[Optim] = []
    for name in names:
        for optim in registry[name]():
            if optim not in optims:  # avoid duplicates
                optims.append(optim)
    if seed is not None:
        np.random.RandomState(seed).shuffle(optims)
    return optims


@registry.register
def large() -> tp.List[Optim]:
    return ["NGO", "Shiwa", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE",
            "OnePlusOne", "SQP", "Cobyla", "Powell",
            "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE",
            "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS", "CM",
            "MultiCMA", "TripleCMA", "MultiScaleCMA", "RSQP", "RCobyla", "RPowell", "SQPCMA", "MetaModel", "PolyCMA", "ManyCMA"]


@registry.register
def cma() -> tp.List[Optim]:
    return ["DiagonalCMA", "CMA"]


@registry.register
def competence_map() -> tp.List[Optim]:
    return ["NGO", "Shiwa"]


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
