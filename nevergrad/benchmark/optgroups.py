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
from nevergrad.optimization.optimizerlib import ConfSplitOptimizer
from nevergrad.optimization.optimizerlib import ParametrizedOnePlusOne

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
def large() -> tp.Sequence[Optim]:
    return ["NGO", "Shiwa", "DiagonalCMA", "CMA", "PSO", "DE", "MiniDE", "QrDE", "MiniQrDE", "LhsDE",
            "OnePlusOne", "SQP", "Cobyla", "Powell",
            "TwoPointsDE", "OnePointDE", "AlmostRotationInvariantDE", "RotationInvariantDE",
            "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "CMandAS", "CM",
            "MultiCMA", "TripleCMA", "MultiScaleCMA", "RSQP", "RCobyla", "RPowell", "SQPCMA", "MetaModel", "PolyCMA", "ManyCMA"]


@registry.register
def emna_variants() -> tp.Sequence[Optim]:
    return ["IsoEMNA", "NaiveIsoEMNA", "AnisoEMNA", "NaiveAnisoEMNA", "CMA", "NaiveTBPSA",
              "NaiveIsoEMNATBPSA", "IsoEMNATBPSA", "NaiveAnisoEMNATBPSA", "AnisoEMNATBPSA"]


@registry.register
def splitters() -> tp.Sequence[Optim]:
    optims:tp.List[Optim] = []
    for num_optims in [None, 3, 5, 9, 13]:
        name = "SplitCMA" + ("Auto" if num_optims is None else str(num_optims))
        opt = ConfSplitOptimizer(
                num_optims=num_optims
            ).set_name(name)
        optims.append(opt)
    return optims


@registry.register
def progressive() -> tp.Sequence[Optim]:
    optims:tp.List[Optim] = []
    for mutation in ["discrete", "gaussian"]:
        for num_optims in [None, 13, 10000]:
            name = "Prog" + ("Disc" if mutation == "discrete" else "") + (
                    "Auto" if num_optims is None else ("Inf" if num_optims == 10000 else str(num_optims)))
            mv = ParametrizedOnePlusOne(noise_handling="optimistic", mutation=mutation)
            opt = ConfSplitOptimizer(
                num_optims=num_optims, progressive=True, multivariate_optimizer=mv
            ).set_name(name)
            optims.append(opt)
    return optims


@registry.register
def basics() -> tp.Sequence[Optim]:
    return ["NGOpt8", "CMandAS2", "CMA", "DE", "MetaModel", "BO"]


@registry.register
def baselines() -> tp.Sequence[Optim]:
    # This list should not change. This is the basics for comparison.
    # No algorithm with unstable other dependency.
    return ["OnePlusOne", "DiscreteOnePlusOne", "NoisyDiscreteOnePlusOne", "PSO", "DE", "TwoPointsDE"]


@registry.register
def parallel_basics() -> tp.Sequence[Optim]:
    return ["NGOpt8", "CMandAS2", "CMA", "DE", "MetaModel"]


@registry.register
def cma() -> tp.Sequence[Optim]:
    return ["DiagonalCMA", "CMA"]


@registry.register
def competence_map() -> tp.Sequence[Optim]:
    return ["NGO", "Shiwa"]


@registry.register
def competitive() -> tp.Sequence[Optim]:
    """A set of competitive algorithms
    """
    return get_optimizers("cma", "competence_map") + ["NaiveTBPSA", "PSO", "DE", "LhsDE", "RandomSearch", "OnePlusOne", "TwoPointsDE"]


@registry.register
def all_bo() -> tp.Sequence[Optim]:
    return sorted(x for x in ng.optimizers.registry if "BO" in x)


@registry.register
def spsa() -> tp.Sequence[Optim]:
    # return sorted(x for x, y in ng.optimizers.registry.items() if (any(e in x for e in "TBPSA SPSA".split()) and "iscr" not in x))
    return ["NaiveTBPSA", "SPSA", "TBPSA"]
