# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import warnings
import typing as tp
import inspect
import itertools
import numpy as np
import nevergrad as ng
import nevergrad.functions.corefuncs as corefuncs
from nevergrad.functions import base as fbase
from nevergrad.functions import ExperimentFunction
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import FarOptimumFunction
from nevergrad.functions.fishing import OptimizeFish
from nevergrad.functions.pbt import PBT
from nevergrad.functions.ml import MLTuning
from nevergrad.functions import mlda as _mlda
from nevergrad.functions.photonics import Photonics
from nevergrad.functions.photonics import ceviche as photonics_ceviche
from nevergrad.functions.arcoating import ARCoating
from nevergrad.functions import images as imagesxp
from nevergrad.functions.powersystems import PowerSystem
from nevergrad.functions.ac import NgAquacrop
from nevergrad.functions.stsp import STSP
from nevergrad.functions.topology_optimization import TO
from nevergrad.functions.lsgo import make_function as lsgo_makefunction
from nevergrad.functions.rocket import Rocket
from nevergrad.functions.mixsimulator import OptimizeMix
from nevergrad.functions.unitcommitment import UnitCommitmentProblem
from nevergrad.functions import control
from nevergrad.functions import rl
from nevergrad.functions.games import game
from nevergrad.functions import iohprofiler
from nevergrad.functions import helpers
from nevergrad.functions.cycling import Cycling
from .xpbase import Experiment as Experiment
from .xpbase import create_seed_generator
from .xpbase import registry as registry  # noqa
from .optgroups import get_optimizers

# register all experiments from other files
# pylint: disable=unused-import
from . import frozenexperiments  # noqa
from . import gymexperiments  # noqa

# pylint: disable=stop-iteration-return, too-many-nested-blocks, too-many-locals


#    list_optims = ["QOTPDE", "LQOTPDE", "LQODE", "BAR4", "NGOpt", "CMandAS2"]
#    list_optims = ["QOTPDE", "LQOTPDE", "LQODE"]
#    list_optims = ["SPQODE", "SQOPSO", "DiagonalCMA"]
def refactor_optims(x: tp.List[tp.Any]) -> tp.List[tp.Any]:  # type: ignore
    # return ["DiscreteLenglerOnePlusOne"]
    #    return ["OLNDiscreteOnePlusOne"]
    # return [np.random.choice([
    #    "NgLn",
    #    "SmallLognormalDiscreteOnePlusOne",
    #    "XLognormalDiscreteOnePlusOne",
    # ])]
    return [
        # "BigLognormalDiscreteOnePlusOne",
        # "DiscreteLenglerOnePlusOne",
        # "NgLn",
        # "SmallLognormalDiscreteOnePlusOne",
        # "XLognormalDiscreteOnePlusOne",
        "XSmallLognormalDiscreteOnePlusOne",
        "MultiLN",
        "NgRS",
        "NgIohRS",
        "NgIohMLn",
        "NgIohLn",
        # "LognormalDiscreteOnePlusOne",
        # "HugeLognormalDiscreteOnePlusOne",
    ]
    # return ["CSEC11"]
    # return [np.random.choice(["CSEC11", "SQOPSODCMA", "NgIoh4", "NGOpt"])]
    # return ["LPCMA"]  #return [np.random.choice(["CSEC10", "DSproba", "NgIoh4", "DSbase", "DS3p", "DSsubspace"])]
    # return x
    # return ["LognormalDiscreteOnePlusOne"]
    # return ["TBPSA", "OptimisticDiscreteOnePlusOne", "NGOpt", "CSEC10"] #CSEC10"]
    # return ["NGOpt", "NgIoh4"]
    #    return ["NgIoh8"]
    # return ["DE", "NGOpt", "NgIoh11", "RandomSearch"]
    # return ["DiscreteLenglerOnePlusOne", "LognormalDiscreteOnePlusOne"]
    # return ["DSproba" + str(i) for i in range(2, 10)]
    # return ["DSproba", "DSsubspace", "DS3p", "DSbase"]
    # return ["DSproba"]
    if False:  # np.random.randn() < 0.0:
        return list(
            np.random.choice(
                [
                    "NgIoh4",
                    "NGOpt",
                    "NGOptRW",
                    "ChainCMASQP",
                    "PymooBIPOP",
                    "NLOPT_LN_SBPLX",
                    "QNDE",
                    "BFGSCMAPlus",
                    "ChainMetaModelSQP",
                    "BFGSCMA",
                    "BAR4",
                    "BFGSCMAPlus",
                    "LBFGSB",
                    "LQOTPDE",
                    "LogSQPCMA",
                ],
                4,
            )
        )
    # return ["RandomSearch", "OnePlusOne", "DE", "PSO"]
    list_optims = x
    algos = {}
    algos["aquacrop_fao"] = [
        "CMA",
        "CMandAS2",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    algos["bonnans"] = [
        "AdaptiveDiscreteOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "DiscreteLenglerFourthOnePlusOne",
        "DiscreteLenglerHalfOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "MemeticDE",
    ]
    algos["double_o_seven"] = [
        "DiagonalCMA",
        "DiscreteDE",
        "MetaTuneRecentering",
        "PSO",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "TBPSA",
    ]
    algos["fishing"] = [
        "CMA",
        "CMandAS2",
        "ChainMetaModelSQP",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    algos["mldakmeans"] = [
        "DE",
        "SplitCMA5",
        "SplitTwoPointsDE3",
        "SplitTwoPointsDE5",
        "SplitTwoPointsDEAuto",
        "TwoPointsDE",
    ]
    algos["mltuning"] = [
        "OnePlusOne",
        "RandomSearch",
    ]
    algos["mono_rocket"] = [
        "CMA",
        "CMandAS2",
        "DE",
        "MetaModel",
        "NGOpt10",
    ]
    algos["ms_bbob"] = [
        "ChainMetaModelSQP",
        "MetaModelOnePlusOne",
        "Powell",
        "QODE",
        "SQP",
        "TinyCMA",
    ]
    algos["multiobjective_example_hd"] = [
        "DiscreteLenglerOnePlusOne",
        "DiscreteOnePlusOne",
        "MetaNGOpt10",
        "ParametrizationDE",
        "RecES",
        "RecMutDE",
    ]
    algos["multiobjective_example_many_hd"] = [
        "DiscreteLenglerOnePlusOne",
        "DiscreteOnePlusOne",
        "MetaNGOpt10",
        "ParametrizationDE",
        "RecES",
        "RecMutDE",
    ]
    algos["multiobjective_example"] = [
        "CMA",
        "DE",
        "ParametrizationDE",
        "RecES",
        "RecMutDE",
    ]
    algos["naive_seq_keras_tuning"] = [
        "CMA",
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["nano_naive_seq_mltuning"] = [
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["nano_seq_mltuning"] = [
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["oneshot_mltuning"] = [
        "DE",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["pbbob"] = [
        "CMAbounded",
        "DE",
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "QODE",
        "QrDE",
    ]
    algos["pbo_reduced_suite"] = [
        "DiscreteLenglerOnePlusOne",
        "LognormalDiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOneT",
        "DiscreteLenglerOnePlusOneT",
        "SADiscreteLenglerOnePlusOneExp09",
        "SADiscreteLenglerOnePlusOneExp09",
        "discretememetic",
    ]
    algos["reduced_yahdlbbbob"] = [
        "CMA",
        "DE",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "PSO",
        "RFMetaModelDE",
    ]
    algos["seq_keras_tuning"] = [
        "CMA",
        "DE",
        "HyperOpt",
        "OnePlusOne",
        "RandomSearch",
        "TwoPointsDE",
    ]
    algos["sequential_topology_optimization"] = [
        "CMA",
        "DE",
        "GeneticDE",
        "OnePlusOne",
        "TwoPointsDE",
        "VoronoiDE",
    ]
    algos["spsa_benchmark"] = [
        "CMA",
        "DE",
        "NaiveTBPSA",
        "OnePlusOne",
        "SPSA",
        "TBPSA",
    ]
    algos["topology_optimization"] = [
        "CMA",
        "DE",
        "GeneticDE",
        "OnePlusOne",
        "TwoPointsDE",
        "VoronoiDE",
    ]
    algos["yabbob"] = [
        "CMA",
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
    ]
    algos["yabigbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "PSO",
        "TwoPointsDE",
    ]
    algos["yaboundedbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaboxbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yamegapenbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yamegapenboundedbbob"] = [
        "CMA",
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yamegapenboxbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yanoisybbob"] = [
        "TBPSA",
        "NoisyRL2",
        "NoisyRL3",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "RBFGS",
        "MicroCMA",
        "NoisyDiscreteOnePlusOne",
        "RandomSearch",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "SQP",
    ]
    algos["yaonepenbbob"] = [
        "CMandAS2",
        "ChainMetaModelSQP",
        "MetaModel",
        "NGOpt",
        "NeuralMetaModel",
        "Shiwa",
    ]
    algos["yaonepenboundedbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaonepenboxbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaonepennoisybbob"] = [
        "NoisyDiscreteOnePlusOne",
        "RandomSearch",
        "SQP",
        "TBPSA",
    ]
    algos["yaonepenparabbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yaonepensmallbbob"] = [
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yaparabbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yapenbbob"] = [
        "ChainMetaModelSQP",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yapenboundedbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yapenboxbbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "OnePlusOne",
        "RFMetaModel",
    ]
    algos["yapennoisybbob"] = [
        "NoisyDiscreteOnePlusOne",
        "RandomSearch",
        "SQP",
        "TBPSA",
    ]
    algos["yapenparabbob"] = [
        "CMA",
        "MetaModel",
        "MetaModelDE",
        "NeuralMetaModel",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yapensmallbbob"] = [
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    algos["yasmallbbob"] = [
        "Cobyla",
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "PSO",
        "RFMetaModelDE",
    ]
    algos["yatinybbob"] = [
        "Cobyla",
        "DE",
        "MetaModel",
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "TwoPointsDE",
    ]
    algos["yatuningbbob"] = [
        "Cobyla",
        "MetaModelOnePlusOne",
        "NeuralMetaModel",
        "RFMetaModelDE",
        "RandomSearch",
        "TwoPointsDE",
    ]

    # Below, we use the best in the records above.
    benchmark = str(inspect.stack()[1].function)
    # if "bbob" in benchmark and np.random.choice([True, False, False, False, False]):
    #    return ["DSproba" + str(i) for i in range(2, 10)]
    if benchmark in algos:  # and np.random.choice([True, False]):  # and np.random.randint(2) > 0 and False:
        list_algos = algos[benchmark][:5] + [
            "CSEC10",
            "NGOpt",
            "NLOPT_LN_SBPLX",
        ]
        return (
            list_algos  # [np.random.choice(list_algos)]
            if (
                "eras" in benchmark
                or "tial_instrum" in benchmark
                or "big" in benchmark
                or "lsgo" in benchmark
                or "rock" in benchmark
            )
            else list_algos  # list(np.random.choice(list_algos, 5))
        )
    if benchmark in algos:
        list_algos = algos[benchmark]
        return (
            list_algos  # [np.random.choice(list_algos)]
            if (
                "eras" in benchmark
                or "tial_instrum" in benchmark
                or "big" in benchmark
                or "lsgo" in benchmark
                or "rock" in benchmark
            )
            else list_algos  # list(np.random.choice(list_algos, 5))
        )
    return [
        "NgDS3",
        "NgIoh4",
        "NgIoh21",
        "NGOpt",
        "NGDSRW",
    ]

    # Here, we pseudo-randomly draw one optim in the provided list,
    # depending on the host (so that each host is using the same optim).
    #    list_optims = x
    #    list_optims = ["BAR", "BAR2", "BAR3"]
    #    list_optims = ["BAR", "BAR2", "BAR3", "BAR4", "NGOpt", "NGOptRW", "CMandAS2"]
    #    list_optims = ["QOTPDE", "LQOTPDE", "LQODE", "BAR4", "NGOpt", "CMandAS2"]
    #    list_optims = ["QOTPDE", "LQOTPDE", "LQODE"]
    #    list_optims = ["SPQODE", "SQOPSO", "DiagonalCMA"]
    #    list_optims = ["BAR", "BAR3", "BAR2", "BAR4", "SPQODE", "SQOPSO", "DiagonalCMA"]
    #    list_optims = ["QODE", "CMA", "SQOPSO", "RandomSearch", "OnePlusOne", "DE"]
    #    list_optims = ["AX", "SMAC3", "pysot"]
    #    # list_optims = ["DiagonalCMA"]
    #    list_optims = ["GeneticDE"]
    #    list_optims = [
    #        "NGOpt",
    #        "CMA",
    #        "DiagonalCMA",
    #        "GeneticDE",
    #        "SQOPSO",
    #        "QODE",
    #        "RandomSearch",
    #        "BFGS",
    #        "PSO",
    #        "DE",
    #        "MetaTuneRecentering",
    #        "MetaRecentering",
    #        "LhsDE",
    #        "HullCenterHullAvgCauchyScrHammersleySearch",
    #    ]
    #    list_optims = [
    #        "QOPSO",
    #        "OnePlusOne",
    #        "NaiveTBPSA",
    #        "LBFGSB",
    #        "LHSSearch",
    #        "DiscreteLenglerOnePlusOneT",
    #        "MetaModel",
    #        "MetaModelOnePlusOne",
    #        "LHSCauchySearch",
    #        "Cobyla",
    #        "CMA",
    #        "DiagonalCMA",
    #    ]
    def doint(s):  # Converting a string into an int.
        return 7 + sum([ord(c) * i for i, c in enumerate(s)])

    import socket

    host = socket.gethostname()

    if "iscr" in benchmark or "pbo" in benchmark:
        list_optims += [
            a
            for a in [
                "DiscreteDE",
                "DiscreteOnePlusOne",
                "SADiscreteLenglerOnePlusOneExp09",
                "SADiscreteLenglerOnePlusOneExp099",
                "SADiscreteLenglerOnePlusOneExp09Auto",
                "SADiscreteLenglerOnePlusOneLinAuto",
                "SADiscreteLenglerOnePlusOneLin1",
                "SADiscreteLenglerOnePlusOneLin100",
                "SADiscreteOnePlusOneExp099",
                "SADiscreteOnePlusOneLin100",
                "SADiscreteOnePlusOneExp09",
                "PortfolioDiscreteOnePlusOne",
                "DiscreteLenglerOnePlusOne",
                "DiscreteLengler2OnePlusOne",
                "DiscreteLengler3OnePlusOne",
                "DiscreteLenglerHalfOnePlusOne",
                "DiscreteLenglerFourthOnePlusOne",
                "AdaptiveDiscreteOnePlusOne",
                "LognormalDiscreteOnePlusOne",
                "AnisotropicAdaptiveDiscreteOnePlusOne",
                "DiscreteBSOOnePlusOne",
                "DiscreteDoerrOnePlusOne",
                "DoubleFastGADiscreteOnePlusOne",
                "SparseDoubleFastGADiscreteOnePlusOne",
                "RecombiningPortfolioDiscreteOnePlusOne",
                "MultiDiscrete",
                "discretememetic",
                "SmoothDiscreteOnePlusOne",
                "SmoothPortfolioDiscreteOnePlusOne",
                "SmoothDiscreteLenglerOnePlusOne",
                "SuperSmoothDiscreteLenglerOnePlusOne",
                "UltraSmoothDiscreteLenglerOnePlusOne",
                "SmoothLognormalDiscreteOnePlusOne",
                "SmoothAdaptiveDiscreteOnePlusOne",
                "SmoothRecombiningPortfolioDiscreteOnePlusOne",
                "SmoothRecombiningDiscreteLanglerOnePlusOne",
                "UltraSmoothRecombiningDiscreteLanglerOnePlusOne",
                "UltraSmoothElitistRecombiningDiscreteLanglerOnePlusOne",
                "SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne",
                "SuperSmoothRecombiningDiscreteLanglerOnePlusOne",
                "SmoothElitistRecombiningDiscreteLanglerOnePlusOne",
                "RecombiningDiscreteLanglerOnePlusOne",
                "DiscreteDE",
                "cGA",
                "NGOpt",
                "NgIoh4",
                "NgIoh5",
                "NgIoh6",
                "NGOptRW",
                "NgIoh7",
            ]
            if ("Smooth" in a or "Lognor" in a or "Recomb" in a)
        ]

    return [list_optims[doint(host) % len(list_optims)]]


#    return x  # ["Zero"] #return x


#    return ["MultiSQP", "MultiCobyla", "MultiBFGS"]
#    return ["NGOpt"]
#    return ["QODE", "QOPSO", "SQOPSO", "QNDE"]
#    return ["MetaTuneRecentering", "MetaRecentering"]
#    return [
#        "NLOPT_LN_SBPLX",
#        "NLOPT_LN_PRAXIS",
#        "NLOPT_GN_DIRECT",
#        "NLOPT_GN_DIRECT_L",
#        "NLOPT_GN_CRS2_LM",
#        "NLOPT_GN_AGS",
#        "NLOPT_GN_ISRES",
#        "NLOPT_GN_ESCH",
#        "NLOPT_LN_COBYLA",
#        "NLOPT_LN_BOBYQA",
#        "NLOPT_LN_NEWUOA_BOUND",
#        "NLOPT_LN_NELDERMEAD",
#    ]
#    return ["LBFGSB"]  # return ["PymooBIPOP"]
# return ["SQPCMA"]
# return ["MetaRecentering"]  # if you want to run only this algorithm
# return random.sample(x, 1)
# return x
# return ["CMandAS2", "Shiwa"]
# return ["RandomSearch", "NaiveTBPSA"]
# return ["SQOPSO", "QODE"]
# return ["QNDE", "MetaModelQODE"]
# return ["SOPSO"]
# return ["QORandomSearch"]


def skip_ci(*, reason: str) -> None:
    """Only use this if there is a good reason for not testing the xp,
    such as very slow for instance (>1min) with no way to make it faster.
    This is dangereous because it won't test reproducibility and the experiment
    may therefore be corrupted with no way to notice it automatically.
    """
    if os.environ.get("NEVERGRAD_PYTEST", False):  # break here for tests
        raise fbase.UnsupportedExperiment("Skipping CI: " + reason)


class _Constraint:
    def __init__(self, name: str, as_bool: bool) -> None:
        self.name = name
        self.as_bool = as_bool

    def __call__(self, data: np.ndarray) -> tp.Union[bool, float]:
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Unexpected inputs as np.ndarray, got {data}")
        if self.name == "sum":
            value = float(np.sum(data))
        elif self.name == "diff":
            value = float(np.sum(data[::2]) - np.sum(data[1::2]))
        elif self.name == "second_diff":
            value = float(2 * np.sum(data[1::2]) - 3 * np.sum(data[::2]))
        elif self.name == "ball":
            # Most points violate the constraint.
            value = float(np.sum(np.square(data)) - len(data) - np.sqrt(len(data)))
        else:
            raise NotImplementedError(f"Unknown function {self.name}")
        return value > 0 if self.as_bool else value


@registry.register
def keras_tuning(
    seed: tp.Optional[int] = None,
    overfitter: bool = False,
    seq: bool = False,
    veryseq: bool = False,
) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on Keras models."""
    seedg = create_seed_generator(seed)
    # Continuous case,

    # First, a few functions with constraints.
    # optims: tp.List[str] = ["PSO", "OnePlusOne"] + get_optimizers("basics", seed=next(seedg))  # type: ignore
    optims = ["OnePlusOne", "BO", "RandomSearch", "CMA", "DE", "TwoPointsDE", "HyperOpt", "PCABO", "Cobyla"]
    optims = [
        "OnePlusOne",
        "RandomSearch",
        "CMA",
        "DE",
        "TwoPointsDE",
        "HyperOpt",
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "RFMetaModel",
        "RFMetaModelOnePlusOne",
    ]
    optims = ["OnePlusOne", "RandomSearch", "Cobyla"]
    optims = ["DE", "TwoPointsDE", "HyperOpt", "MetaModelOnePlusOne"]
    optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    optims = [
        "MetaTuneRecentering",
        "MetaRecentering",
        "HullCenterHullAvgCauchyScrHammersleySearch",
        "LHSSearch",
        "LHSCauchySearch",
    ]
    optims = ["NGOpt", "NGOptRW", "QODE"]
    optims = ["NGOpt"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = refactor_optims(optims)
    datasets = ["kerasBoston", "diabetes", "auto-mpg", "red-wine", "white-wine"]
    optims = refactor_optims(optims)
    for dimension in [None]:
        for dataset in datasets:
            function = MLTuning(
                regressor="keras_dense_nn", data_dimension=dimension, dataset=dataset, overfitter=overfitter
            )
            for budget in [150, 500]:
                for num_workers in (
                    [1, budget // 4] if seq else [budget]
                ):  # Seq for sequential optimization experiments.
                    if veryseq and num_workers > 1:
                        continue
                    for optim in optims:
                        xp = Experiment(
                            function, optim, num_workers=num_workers, budget=budget, seed=next(seedg)
                        )
                        skip_ci(reason="too slow")
                        xp.function.parametrization.real_world = True
                        xp.function.parametrization.hptuning = True
                        if not xp.is_incoherent:  # and np.random.choice([True, False]):
                            yield xp


@registry.register
def mltuning(
    seed: tp.Optional[int] = None,
    overfitter: bool = False,
    seq: bool = False,
    veryseq: bool = False,
    nano: bool = False,
) -> tp.Iterator[Experiment]:
    """Machine learning hyperparameter tuning experiment. Based on scikit models."""
    seedg = create_seed_generator(seed)
    # optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore
    # if not seq:
    #    optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    optims = ["OnePlusOne", "BO", "RandomSearch", "CMA", "DE", "TwoPointsDE", "PCABO", "HyperOpt", "Cobyla"]
    optims = [
        "OnePlusOne",
        "RandomSearch",
        "CMA",
        "DE",
        "TwoPointsDE",
        "HyperOpt",
        "Cobyla",
        "MetaModel",
        "MetaModelOnePlusOne",
        "RFMetaModel",
        "RFMetaModelOnePlusOne",
    ]
    optims = ["OnePlusOne", "RandomSearch", "Cobyla"]
    optims = ["DE", "TwoPointsDE", "HyperOpt", "MetaModelOnePlusOne"]
    optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    optims = [
        "MetaTuneRecentering",
        "MetaRecentering",
        "HullCenterHullAvgCauchyScrHammersleySearch",
        "LHSSearch",
        "LHSCauchySearch",
    ]
    optims = ["NGOpt", "NGOptRW", "QODE"]
    optims = ["NGOpt"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = refactor_optims(optims)
    for dimension in [None, 1, 2, 3]:
        if dimension is None:
            datasets = ["diabetes", "auto-mpg", "red-wine", "white-wine"]
        else:
            datasets = ["artificialcos", "artificial", "artificialsquare"]
        for regressor in ["mlp", "decision_tree", "decision_tree_depth"]:
            for dataset in datasets:
                function = MLTuning(
                    regressor=regressor, data_dimension=dimension, dataset=dataset, overfitter=overfitter
                )
                for budget in [150, 500] if not nano else [80, 160]:
                    # Seq for sequential optimization experiments.
                    parallelization = [1, budget // 4] if seq else [budget]
                    for num_workers in parallelization:
                        if veryseq and num_workers > 1:
                            continue

                        for optim in optims:
                            xp = Experiment(
                                function, optim, num_workers=num_workers, budget=budget, seed=next(seedg)
                            )
                            skip_ci(reason="too slow")
                            xp.function.parametrization.real_world = True
                            xp.function.parametrization.hptuning = True
                            if not xp.is_incoherent:  # and np.random.choice([True, False]):
                                yield xp


@registry.register
def naivemltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True)


@registry.register
def veryseq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of keras tuning."""
    return keras_tuning(seed, overfitter=False, seq=True, veryseq=True)


@registry.register
def seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of keras tuning."""
    return keras_tuning(seed, overfitter=False, seq=True)


@registry.register
def naive_seq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Naive counterpart (no overfitting, see naivemltuning)of seq_keras_tuning."""
    return keras_tuning(seed, overfitter=True, seq=True)


@registry.register
def naive_veryseq_keras_tuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Naive counterpart (no overfitting, see naivemltuning)of seq_keras_tuning."""
    return keras_tuning(seed, overfitter=True, seq=True, veryseq=True)


@registry.register
def oneshot_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One-shot counterpart of Scikit tuning."""
    return mltuning(seed, overfitter=False, seq=False)


# We register only the (almost) sequential counterparts for the moment.
@registry.register
def seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning."""
    return mltuning(seed, overfitter=False, seq=True)


@registry.register
def nano_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of seq_mltuning with smaller budget."""
    return mltuning(seed, overfitter=False, seq=True, nano=True)


@registry.register
def nano_veryseq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of seq_mltuning with smaller budget."""
    return mltuning(seed, overfitter=False, seq=True, nano=True, veryseq=True)


@registry.register
def nano_naive_veryseq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test,
    and with lower budget."""
    return mltuning(seed, overfitter=True, seq=True, nano=True, veryseq=True)


@registry.register
def nano_naive_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test,
    and with lower budget."""
    return mltuning(seed, overfitter=True, seq=True, nano=True)


@registry.register
def naive_seq_mltuning(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Iterative counterpart of mltuning with overfitting of valid loss, i.e. train/valid/valid instead of train/valid/test."""
    return mltuning(seed, overfitter=True, seq=True)


# pylint:disable=too-many-branches
@registry.register
def yawidebbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Yet Another Wide Black-Box Optimization Benchmark.
    The goal is basically to have a very wide family of problems: continuous and discrete,
    noisy and noise-free, mono- and multi-objective,  constrained and not constrained, sequential
    and parallel.

    TODO(oteytaud): this requires a significant improvement, covering mixed problems and different types of constraints.
    """
    seedg = create_seed_generator(seed)
    total_xp_per_optim = 0
    # Continuous case

    # First, a few functions with constraints.
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation, translation_factor=tf)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
        for tf in [0.1, 10.0]
    ]
    for i, func in enumerate(functions):
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=i % 2 == 0))
    assert len(functions) == 8
    # Then, let us build a constraint-free case. We include the noisy case.
    names = ["hm", "rastrigin", "sphere", "doublelinearslope", "ellipsoid"]

    functions += [
        ArtificialFunction(
            name,
            block_dimension=d,
            rotation=rotation,
            noise_level=nl,
            split=split,
            translation_factor=tf,
            num_blocks=num_blocks,
        )
        for name in names  # period 5
        for rotation in [True, False]  # period 2
        for nl in [0.0, 100.0]  # period 2
        for tf in [0.1, 10.0]
        for num_blocks in [1, 8]  # period 2
        for d in [5, 70, 10000]  # period 4
        for split in [True, False]  # period 2
    ][
        ::37
    ]  # 37 is coprime with all periods above so we sample correctly the possibilities.
    assert len(functions) == 21, f"{len(functions)} problems instead of 21. Yawidebbob should be standard."
    # This problem is intended as a stable basis forever.
    # The list of optimizers should contain only the basic for comparison and "baselines".
    # optims: tp.List[str] = ["NGOpt10"] + get_optimizers("baselines", seed=next(seedg))  # type: ignore
    optims = ["NGOptRW", "NGOpt", "RandomSearch", "CMA", "DE", "DiscreteLenglerOnePlusOne"]
    optims = refactor_optims(optims)
    # optims = optims[:2]
    index = 0
    for function in functions:
        for budget in [50, 1500, 25000]:
            for nw in [1, budget] + ([] if budget <= 300 else [300]):
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        xp = Experiment(function, optim, num_workers=nw, budget=budget, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp

    assert total_xp_per_optim == 33, f"We have 33 single-obj xps per optimizer (got {total_xp_per_optim})."
    # Discrete, unordered.
    index = 0
    for nv in [200, 2000]:
        for arity in [2, 7, 37]:
            instrum = ng.p.TransitionChoice(range(arity), repetitions=nv)
            for name in ["onemax", "leadingones", "jump"]:
                index += 1
                if index % 4 != 0:
                    continue
                dfunc = ExperimentFunction(
                    corefuncs.DiscreteFunction(name, arity), instrum.set_name("transition")
                )
                dfunc.add_descriptors(arity=arity)
                for budget in [500, 1500, 5000]:
                    for nw in [1, 100]:
                        total_xp_per_optim += 1
                        for optim in optims:
                            yield Experiment(dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg))
    assert (
        total_xp_per_optim == 57
    ), f"Including discrete, we check xps per optimizer (got {total_xp_per_optim})."

    # The multiobjective case.
    # TODO the upper bounds are really not well set for this experiment with cigar
    mofuncs: tp.List[fbase.MultiExperiment] = []
    for name1 in ["sphere", "ellipsoid"]:
        for name2 in ["sphere", "hm"]:
            for tf in [0.25, 4.0]:
                mofuncs += [
                    fbase.MultiExperiment(
                        [
                            ArtificialFunction(name1, block_dimension=7),
                            ArtificialFunction(name2, block_dimension=7, translation_factor=tf),
                        ],
                        upper_bounds=np.array((100.0, 100.0)),
                    )
                ]
                mofuncs[-1].add_descriptors(num_objectives=2)
    for name1 in ["sphere", "ellipsoid"]:
        for name2 in ["sphere", "hm"]:
            for name3 in ["sphere", "hm"]:
                for tf in [0.25, 4.0]:
                    mofuncs += [
                        fbase.MultiExperiment(
                            [
                                ArtificialFunction(name1, block_dimension=7, translation_factor=1.0 / tf),
                                ArtificialFunction(name2, block_dimension=7, translation_factor=tf),
                                ArtificialFunction(name3, block_dimension=7),
                            ],
                            upper_bounds=np.array((100.0, 100.0, 100.0)),
                        )
                    ]
                    mofuncs[-1].add_descriptors(num_objectives=3)
    index = 0
    for mofunc in mofuncs[::3]:
        for budget in [2000, 4000, 8000]:
            for nw in [1, 20, 100]:
                index += 1
                if index % 5 == 0:
                    total_xp_per_optim += 1
                    for optim in optims:
                        yield Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
    assert total_xp_per_optim == 71, f"We should have 71 xps per optimizer, not {total_xp_per_optim}."


# pylint: disable=redefined-outer-name
@registry.register
def parallel_small_budget(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization with small budgets"""
    seedg = create_seed_generator(seed)
    # optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore
    optims = ["DE", "TwoPointsDE", "CMA", "NGOpt", "PSO", "OnePlusOne", "RandomSearch"]
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "multipeak"]
    names += ["sphere", "cigar", "ellipsoid", "altellipsoid"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # funcs
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation)
        for name in names
        for rotation in [True, False]
        for d in [2, 4, 8]
    ]
    budgets = [10, 50, 100, 200, 400]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                for nw in [2, 8, 16]:
                    for batch in [True, False]:
                        if nw < budget / 4:
                            xp = Experiment(
                                function,
                                optim,
                                num_workers=nw,
                                budget=budget,
                                batch_mode=batch,
                                seed=next(seedg),
                            )
                            if not xp.is_incoherent:
                                yield xp


@registry.register
def instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Comparison of optimization algorithms equipped with distinct instrumentations.
    Onemax, Leadingones, Jump function."""
    # Discrete, unordered.

    seedg = create_seed_generator(seed)
    # optims = get_optimizers("small_discrete", seed=next(seedg))
    optims = ["DiscreteOnePlusOne", "NGOpt", "CMA", "TwoPointsDE", "DiscreteLenglerOnePlusOne"]
    optims = ["RFMetaModelOnePlusOne"]
    optims = ["FastGADiscreteOnePlusOne"]
    optims = ["DoubleFastGADiscreteOnePlusOne"]
    optims = ["DiscreteOnePlusOne"]
    optims = ["OnePlusOne"]
    optims = ["DiscreteLenglerOnePlusOne"]
    optims = ["NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum: ng.p.Parameter = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    assert instrum_str in ("Ordered", "Unordered")
                    instrum = ng.p.TransitionChoice(
                        range(arity), repetitions=nv, ordered=instrum_str == "Ordered"
                    )
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(
                        corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str)
                    )
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
                    dfunc.add_descriptors(instrum_str=instrum_str)
                    for optim in optims:
                        for nw in [1, 10]:
                            for budget in [50, 500, 5000]:
                                if np.random.rand() > 0.3:
                                    continue
                                yield Experiment(
                                    dfunc, optim, num_workers=nw, budget=budget, seed=next(seedg)
                                )


@registry.register
def sequential_instrum_discrete(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of instrum_discrete."""

    seedg = create_seed_generator(seed)
    # Discrete, unordered.
    # optims = get_optimizers("discrete", seed=next(seedg))
    optims = ["DiscreteOnePlusOne", "NGOpt", "CMA", "TwoPointsDE", "DiscreteLenglerOnePlusOne"]
    optims = ["OnePlusOne"]
    optims = ["DiscreteLenglerOnePlusOne"]
    optims = ["NGOpt", "NGOptRW"]
    optims = [
        l
        for l in list(ng.optimizers.registry.keys())
        if "DiscreteOneP" in l
        and "SA" not in l
        and "Smooth" not in l
        and "Noisy" not in l
        and "Optimis" not in l
        and "T" != l[-1]
    ] + ["cGA", "DiscreteDE"]
    optims = refactor_optims(optims)
    for nv in [10, 50, 200, 1000, 5000]:
        for arity in [2, 3, 7, 30]:
            for instrum_str in ["Unordered", "Softmax", "Ordered"]:
                if instrum_str == "Softmax":
                    instrum: ng.p.Parameter = ng.p.Choice(range(arity), repetitions=nv)
                else:
                    instrum = ng.p.TransitionChoice(
                        range(arity), repetitions=nv, ordered=instrum_str == "Ordered"
                    )
                for name in ["onemax", "leadingones", "jump"]:
                    dfunc = ExperimentFunction(
                        corefuncs.DiscreteFunction(name, arity), instrum.set_name(instrum_str)
                    )
                    dfunc.add_descriptors(arity=arity)
                    dfunc.add_descriptors(nv=nv)
                    dfunc.add_descriptors(instrum_str=instrum_str)
                    for optim in optims:
                        for budget in [50, 500, 5000, 50000]:
                            yield Experiment(dfunc, optim, budget=budget, seed=next(seedg))


@registry.register
def deceptive(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Very difficult objective functions: one is highly multimodal (infinitely many local optima),
    one has an infinite condition number, one has an infinitely long path towards the optimum.
    Looks somehow fractal."""
    seedg = create_seed_generator(seed)
    names = ["deceptivemultimodal", "deceptiveillcond", "deceptivepath"]
    optims = get_optimizers("basics", seed=next(seedg))
    optims = ["CMA", "DE", "TwoPointsDE", "PSO", "OnePlusOne", "RandomSearch", "NGOptRW"]
    optims = [
        "RBFGS",
        "LBFGSB",
        "DE",
        "TwoPointsDE",
        "RandomSearch",
        "OnePlusOne",
        "PSO",
        "CMA",
        "ChainMetaModelSQP",
        "MemeticDE",
        "MetaModel",
        "RFMetaModel",
        "MetaModelDE",
        "RFMetaModelDE",
    ]
    optims = ["NGOpt"]
    functions = [
        ArtificialFunction(
            name, block_dimension=2, num_blocks=n_blocks, rotation=rotation, aggregator=aggregator
        )
        for name in names
        for rotation in [False, True]
        for n_blocks in [1, 2, 8, 16]
        for aggregator in ["sum", "max"]
    ]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [
                25,
                37,
                50,
                75,
                87,
                100,
                200,
                400,
                800,
                1600,
                3200,
                6400,
                12800,
            ]:  # + list(range(100, 20001, 500)):
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def lowbudget(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims: tp.List[str] = [
        "AX",
        "BOBYQA",
        "Cobyla",
        "RandomSearch",
        "CMA",
        "NGOpt",
        "DE",
        "PSO",
        "pysot",
        "negpysot",
    ]
    functions = [
        ArtificialFunction(name, block_dimension=bd, bounded=b)
        for name in names
        for bd in [7]
        for b in [True, False]
    ]
    for func in functions:
        for optim in optims:
            for budget in [10, 20, 30]:
                yield Experiment(func, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def parallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization on 3 classical objective functions: sphere, rastrigin, cigar.
    The number of workers is 20 % of the budget.
    Testing both no useless variables and 5/6 of useless variables."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims: tp.List[str] = get_optimizers("parallel_basics", seed=next(seedg))  # type: ignore
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [25]
        for uv_factor in [0, 5]
    ]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=int(budget / 5), seed=next(seedg))


@registry.register
def harderparallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization on 4 classical objective functions. More distinct settings than << parallel >>."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar", "ellipsoid"]
    optims = ["NGOpt10"] + get_optimizers("emna_variants", seed=next(seedg))  # type: ignore
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [5, 25]
        for uv_factor in [0, 5]
    ]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000]:
                for num_workers in [int(budget / 10), int(budget / 5), int(budget / 3)]:
                    yield Experiment(func, optim, budget=budget, num_workers=num_workers, seed=next(seedg))


@registry.register
def oneshot(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    0 or 5 dummy variables per real variable.
    Base dimension 3 or 25.
    budget 30, 100 or 3000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 10, 30, 100, 300, 1000, 3000]
        for uv_factor in [0]  # , 5]
    ]
    for func in functions:
        for optim in optims:
            # if not any(x in str(optim) for x in ["Tune", "Large", "Cauchy"]):
            # if "Meta" in str(optim):
            for budget in [100000, 30, 100, 300, 1000, 3000, 10000]:
                if func.dimension < 3000 or budget < 100000:
                    yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def doe(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Base dimension 2000 or 20000. No rotation, no dummy variable.
    Budget 30, 100, 3000, 10000, 30000, 100000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [2000, 20000]  # 3, 10, 25, 200, 2000]
        for uv_factor in [0]
    ]
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000, 10000, 30000, 100000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def newdoe(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """One shot optimization of 3 classical objective functions (sphere, rastrigin, cigar), simplified.
    Tested on more dimensionalities than doe, namely 20, 200, 2000, 20000. No dummy variables.
    Budgets 30, 100, 3000, 10000, 30000, 100000, 300000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [2000, 20, 200, 20000]  # 3, 10, 25, 200, 2000]
        for uv_factor in [0]
    ]
    budgets = [30, 100, 3000, 10000, 30000, 100000, 300000]
    for func in functions:
        for optim in optims:
            for budget in budgets:
                yield Experiment(func, optim, budget=budget, num_workers=budget, seed=next(seedg))


@registry.register
def fiveshots(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Five-shots optimization of 3 classical objective functions (sphere, rastrigin, cigar).
    Base dimension 3 or 25. 0 or 5 dummy variable per real variable. Budget 30, 100 or 3000."""
    seedg = create_seed_generator(seed)
    names = ["sphere", "rastrigin", "cigar"]
    optims = get_optimizers("oneshot", "basics", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [30, 100, 3000]:
                yield Experiment(func, optim, budget=budget, num_workers=budget // 5, seed=next(seedg))


@registry.register
def multimodal(seed: tp.Optional[int] = None, para: bool = False) -> tp.Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal.
    0 or 5 dummy variable per real variable.
    Base dimension 3 or 25.
    Budget in 3000, 10000, 30000, 100000.
    Sequential.
    """
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.
    optims = get_optimizers("basics", seed=next(seedg))
    if not para:
        optims += get_optimizers("scipy", seed=next(seedg))
    optims = [
        "RBFGS",
        "LBFGSB",
        "DE",
        "TwoPointsDE",
        "RandomSearch",
        "OnePlusOne",
        "PSO",
        "CMA",
        "ChainMetaModelSQP",
        "MemeticDE",
        "MetaModel",
        "RFMetaModel",
        "MetaModelDE",
        "RFMetaModelDE",
    ]
    # + list(sorted(x for x, y in ng.optimizers.registry.items() if "Chain" in x or "BO" in x))
    optims = ["NGOpt"]
    functions = [
        ArtificialFunction(name, block_dimension=bd, useless_variables=bd * uv_factor)
        for name in names
        for bd in [3, 25]
        for uv_factor in [0, 5]
    ]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000, 30000, 100000]:
                for nw in [1000] if para else [1]:
                    xp = Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def hdmultimodal(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Experiment on multimodal functions, namely hm, rastrigin, griewank, rosenbrock, ackley, lunacek,
    deceptivemultimodal. Similar to multimodal, but dimension 20 or 100 or 1000. Budget 1000 or 10000, sequential.
    """
    seedg = create_seed_generator(seed)
    names = ["hm", "rastrigin", "griewank", "rosenbrock", "ackley", "lunacek", "deceptivemultimodal"]
    # Keep in mind that Rosenbrock is multimodal in high dimension http://ieeexplore.ieee.org/document/6792472/.

    optims = get_optimizers("basics", "multimodal", seed=next(seedg))
    optims = [
        "RBFGS",
        "LBFGSB",
        "DE",
        "TwoPointsDE",
        "RandomSearch",
        "OnePlusOne",
        "PSO",
        "CMA",
        "ChainMetaModelSQP",
        "MemeticDE",
        "MetaModel",
        "RFMetaModel",
        "MetaModelDE",
        "RFMetaModelDE",
    ]
    functions = [
        ArtificialFunction(name, block_dimension=bd)
        for name in names
        for bd in [
            1000,
            6000,
            36000,
        ]  # This has been modified, given that it was not sufficiently high-dimensional for its name.
    ]
    optims = refactor_optims(optims)
    for func in functions:
        for optim in optims:
            for budget in [3000, 10000]:
                for nw in [1]:
                    yield Experiment(func, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def paramultimodal(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel counterpart of the multimodal experiment: 1000 workers."""
    return multimodal(seed, para=True)


@registry.register
def bonnans(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.TransitionChoice(range(2), repetitions=100, ordered=False)
    softmax_instrum: ng.p.Parameter = ng.p.Choice(range(2), repetitions=100)
    optims = [
        "RotatedTwoPointsDE",
        "DiscreteLenglerOnePlusOne",
        "DiscreteLengler2OnePlusOne",
        "DiscreteLengler3OnePlusOne",
        "DiscreteLenglerHalfOnePlusOne",
        "DiscreteLenglerFourthOnePlusOne",
        "PortfolioDiscreteOnePlusOne",
        "FastGADiscreteOnePlusOne",
        "DiscreteDoerrOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "DiscreteOnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "GeneticDE",
        "DE",
        "TwoPointsDE",
        "DiscreteOnePlusOne",
        "CMA",
        "SQP",
        "MetaModel",
        "DiagonalCMA",
    ]
    optims = ["RFMetaModelOnePlusOne"]
    optims = ["MemeticDE", "cGA", "DoubleFastGADiscreteOnePlusOne", "FastGADiscreteOnePlusOne"]
    optims = ["NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for i in range(21):
        bonnans = corefuncs.BonnansFunction(index=i)
        for optim in optims:
            instrum_str = "TransitionChoice" if "Discrete" in optim else "Softmax"
            dfunc = ExperimentFunction(
                bonnans,
                instrum.set_name("") if instrum_str == "TransitionChoice" else softmax_instrum.set_name(""),
            )
            dfunc.add_descriptors(index=i)
            dfunc.add_descriptors(instrum_str=instrum_str)
            # dfunc._descriptors = {'index': i}
            for budget in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
                yield Experiment(dfunc, optim, num_workers=1, budget=budget, seed=next(seedg))


# pylint: disable=redefined-outer-name,too-many-arguments
@registry.register
def yabbob(
    seed: tp.Optional[int] = None,
    parallel: bool = False,
    big: bool = False,
    small: bool = False,
    noise: bool = False,
    hd: bool = False,
    constraint_case: int = 0,  # Positive for cheap_constraint, negative for penalized constraints
    split: bool = False,
    tuning: bool = False,
    reduction_factor: int = 1,
    bounded: bool = False,
    box: bool = False,
    max_num_constraints: int = 4,
    mega_smooth_penalization: int = 0,
) -> tp.Iterator[Experiment]:
    """Yet Another Black-Box Optimization Benchmark.
    Related to, but without special effort for exactly sticking to, the BBOB/COCO dataset.
    Dimension 2, 10 and 50.
    Budget 50, 200, 800, 3200, 12800.
    Both rotated or not rotated.
    """
    seedg = create_seed_generator(seed)

    # List of objective functions.
    names = [
        "hm",
        "rastrigin",
        "griewank",
        "rosenbrock",
        "ackley",
        "lunacek",
        "deceptivemultimodal",
        "bucherastrigin",
        "multipeak",
    ]
    names += ["sphere", "doublelinearslope", "stepdoublelinearslope"]
    names += ["cigar", "altcigar", "ellipsoid", "altellipsoid", "stepellipsoid", "discus", "bentcigar"]
    names += ["deceptiveillcond", "deceptivemultimodal", "deceptivepath"]
    # Deceptive path is related to the sharp ridge function; there is a long path to the optimum.
    # Deceptive illcond is related to the difference of powers function; the conditioning varies as we get closer to the optimum.
    # Deceptive multimodal is related to the Weierstrass function and to the Schaffers function.

    # Parametrizing the noise level.
    if noise:
        noise_level = 100000 if hd else 100
    else:
        noise_level = 0

    # Choosing the list of optimizers.
    # optims: tp.List[str] = get_optimizers("competitive", seed=next(seedg))  # type: ignore
    optims = [
        "OnePlusOne",
        "MetaModel",
        "CMA",
        "DE",
        "PSO",
        "TwoPointsDE",
        "RandomSearch",
        "ChainMetaModelSQP",
        "NeuralMetaModel",
        "MetaModelDE",
        "MetaModelOnePlusOne",
    ]
    if noise:
        optims += ["TBPSA", "SQP", "NoisyDiscreteOnePlusOne"]
    if hd:
        optims += ["OnePlusOne"]
        optims += get_optimizers("splitters", seed=next(seedg))  # type: ignore

    if hd and small:
        optims += ["BO", "PCABO", "CMA", "PSO", "DE"]
    if small and not hd:
        optims += ["PCABO", "BO", "Cobyla"]
    optims = [
        "MetaModelDE",
        "MetaModelOnePlusOne",
        "OnePlusOne",
        "ChainMetaModelSQP",
        "RFMetaModel",
        "RFMetaModelDE",
    ]
    # if bounded:
    #    optims = ["BO", "PCABO", "BayesOptimBO", "CMA", "PSO", "DE"]
    # if box:
    #    optims = ["DiagonalCMA", "Cobyla", "NGOpt16", "NGOpt15", "CMandAS2", "OnePlusOne"]
    # List of objective functions.
    optims = [
        "MetaModelDE",
        "NeuralMetaModelDE",
        "SVMMetaModelDE",
        "RFMetaModelDE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModelTwoPointsDE",
        "SVMMetaModelTwoPointsDE",
        "RFMetaModelTwoPointsDE",
        "GeneticDE",
    ]
    optims = ["LargeCMA", "TinyCMA", "OldCMA", "MicroCMA"]
    optims = ["RBFGS", "LBFGSB"]
    optims = get_optimizers("oneshot", seed=next(seedg))  # type: ignore
    optims = [
        "MetaTuneRecentering",
        "MetaRecentering",
        "HullCenterHullAvgCauchyScrHammersleySearch",
        "LHSSearch",
        "LHSCauchySearch",
    ]
    optims = [
        "RBFGS",
        "LBFGSB",
        "MicroCMA",
        "RandomSearch",
        "NoisyDiscreteOnePlusOne",
        "TBPSA",
        "TinyCMA",
        "CMA",
        "ChainMetaModelSQP",
        "OnePlusOne",
        "MetaModel",
        "RFMetaModel",
        "DE",
    ]
    optims = ["NGOpt", "NGOptRW"]
    optims = ["QrDE", "QODE", "LhsDE"]
    optims = ["NGOptRW"]
    if noise:
        optims = [
            #        "MicroCMA",
            #        "TinyCMA",
            #            "SQP",
            #            "NoisyDiscreteOnePlusOne",
            #            "TBPSA",
            #        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
            #
            #            "CMA",
            #            "TinyCMA",
            # "LPCMA",
            # "VLPCMA",
            # "MetaTuneRecentering",
            # "MetaRecentering",
            # "SPSA",
            # "TinySQP",
            # "MicroSQP",
            # "TinySPSA",
            # "MetaModel",
            # "RFMetaModel",
            # "RFMetaModelOnePlusOne",
            "NoisyOnePlusOne",
            # "MicroSPSA",
        ]
    else:
        optims = ["MetaModelPSO", "RFMetaModelPSO", "SVMMetaModelPSO"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["NGOpt"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    functions = [
        ArtificialFunction(
            name,
            block_dimension=d,
            rotation=rotation,
            noise_level=noise_level,
            split=split,
            num_blocks=num_blocks,
            bounded=bounded or box,
        )
        for name in names
        for rotation in [True, False]
        for num_blocks in ([1] if not split else [7, 12])
        for d in (
            [100, 1000, 3000]
            if hd
            else (
                [2, 5, 10, 15]
                if tuning
                else ([40] if bounded else ([2, 3, 5, 10, 15, 20, 50] if noise else [2, 10, 50]))
            )
        )
    ]

    assert reduction_factor in [1, 7, 13, 17]  # needs to be a cofactor
    functions = functions[::reduction_factor]

    # We possibly add constraints.
    constraints: tp.List[tp.Any] = [
        _Constraint(name, as_bool)
        for as_bool in [False, True]
        for name in ["sum", "diff", "second_diff", "ball"]
    ]
    if mega_smooth_penalization > 0:
        constraints = []
        dim = 1000
        max_num_constraints = mega_smooth_penalization
        constraint_case = -abs(constraint_case)
        # We organize constraints so that xs satisfied all of them
        xs = np.random.rand(dim)

        def make_ctr(i):
            xfail = np.random.RandomState(i).rand(dim)

            def f(x):
                local_dim = min(dim, len(x))
                x = x[:local_dim]
                normal = np.exp(np.random.RandomState(i + 31721).randn() - 1.0) * np.linalg.norm(
                    (x - xs[:local_dim]) * np.random.RandomState(i + 741).randn(local_dim)
                )
                return normal - np.sum(
                    (xs[:local_dim] - xfail[:local_dim]) * (x - (xs[:local_dim] + xfail[:local_dim]) / 2.0)
                )

            return f

        for i in range(mega_smooth_penalization):
            f = make_ctr(i)
            assert f(xs) <= 0.0
            constraints += [f]
    assert (
        abs(constraint_case) < len(constraints) + max_num_constraints
    ), "abs(constraint_case) should be in 0, 1, ..., {len(constraints) + max_num_constraints - 1} (0 = no constraint)."
    # We reduce the number of tests when there are constraints, as the number of cases
    # is already multiplied by the number of constraint_case.
    for func in functions[:: 13 if abs(constraint_case) > 0 else 1]:
        func.constraint_violation = []
        # We add a window of the list of constraints. This windows finishes at "constraints" (hence, is empty if
        # constraint_case=0).
        for constraint in constraints[
            max(0, abs(constraint_case) - max_num_constraints) : abs(constraint_case)
        ]:
            if constraint_case > 0:
                func.parametrization.register_cheap_constraint(constraint)
            elif constraint_case < 0:
                func.constraint_violation += [
                    constraint
                ]  # Just for storing, we will move it to the experiment soon

    budgets = (
        [40000, 80000, 160000, 320000]
        if (big and not noise)
        else ([50, 200, 800, 3200, 12800] if not noise else [3200, 12800, 51200, 102400])
    )
    if small and not noise:
        budgets = [10, 20, 40]
    if bounded:
        budgets = [10, 20, 40, 100, 300]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in budgets:
                xp = Experiment(
                    function,
                    optim,
                    num_workers=100 if parallel else 1,
                    budget=budget,
                    seed=next(seedg),
                    constraint_violation=function.constraint_violation,
                )
                if constraint_case != 0:
                    xp.function.parametrization.has_constraints = True
                if np.random.rand() > 0.25:
                    continue
                if not xp.is_incoherent:
                    yield xp


@registry.register
def yahdlbbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with HD and low budget."""
    return yabbob(seed, hd=True, small=True)


@registry.register
def reduced_yahdlbbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with HD and low budget."""
    return yabbob(seed, hd=True, small=True, reduction_factor=17)


@registry.register
def yanoisysplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, noise=True, parallel=False, split=True)


@registry.register
def yahdnoisysplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, hd=True, noise=True, parallel=False, split=True)


@registry.register
def yaconstrainedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with constraints. Constraints are cheap: we do not count calls to them."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=i) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yapenbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=-i) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yamegapenhdbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    slices = [yabbob(seed, hd=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)


@registry.register
def yaonepenbigbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    slices = [yabbob(seed, big=True, constraint_case=-i, max_num_constraints=1) for i in range(1, 7)]
    return itertools.chain(*slices)


@registry.register
def yamegapenbigbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    slices = [yabbob(seed, big=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)


@registry.register
def yamegapenboxbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    slices = [yabbob(seed, box=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)


@registry.register
def yamegapenbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    slices = [yabbob(seed, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)]
    return itertools.chain(*slices)


@registry.register
def yamegapenboundedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    slices = [
        yabbob(seed, bounded=True, constraint_case=-1, mega_smooth_penalization=1000) for i in range(1, 7)
    ]
    return itertools.chain(*slices)


@registry.register
def yapensmallbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yasmallbbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=-i, small=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yapenboundedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabooundedbbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=-i, bounded=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yapennoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yanoisybbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=-i, noise=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yapenparabbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yaparabbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=-i, parallel=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yapenboxbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yaboxbbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, constraint_case=-i, box=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yaonepenbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yaonepensmallbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yasmallbbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, small=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yaonepenboundedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabooundedbbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, bounded=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yaonepennoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yanoisybbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, noise=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yaonepenparabbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yaparabbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, parallel=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yaonepenboxbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yaboxbbob with penalized constraints."""
    cases = 8  # total number of cases (skip 0, as it's constraint-free)
    slices = [yabbob(seed, max_num_constraints=1, constraint_case=-i, box=True) for i in range(1, cases)]
    return itertools.chain(*slices)


@registry.register
def yahdnoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True, noise=True)


@registry.register
def yabigbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with more budget."""
    return yabbob(seed, parallel=False, big=True)


@registry.register
def yasplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with splitting info in the instrumentation."""
    return yabbob(seed, parallel=False, split=True)


@registry.register
def yahdsplitbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yasplitbbob with more dimension."""
    return yabbob(seed, hd=True, split=True)


@registry.register
def yatuningbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget and less dimension."""
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13, tuning=True)


@registry.register
def yatinybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget and less xps."""
    return yabbob(seed, parallel=False, big=False, small=True, reduction_factor=13)


@registry.register
def yasmallbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with less budget."""
    return yabbob(seed, parallel=False, big=False, small=True)


@registry.register
def yahdbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return yabbob(seed, hd=True)


@registry.register
def yaparabbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Parallel optimization counterpart of yabbob."""
    return yabbob(seed, parallel=True, big=False)


@registry.register
def yanoisybbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization counterpart of yabbob.
    This is supposed to be consistent with normal practices in noisy
    optimization: we distinguish recommendations and exploration.
    This is different from the original BBOB/COCO from that point of view.
    """
    return yabbob(seed, noise=True)


@registry.register
def yaboundedbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with bounded domain and dim only 40, (-5,5)**n by default."""
    return yabbob(seed, bounded=True)


@registry.register
def yaboxbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with bounded domain, (-5,5)**n by default."""
    return yabbob(seed, box=True)


@registry.register
def ms_bbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on exponentiated problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """

    seedg = create_seed_generator(seed)
    optims = [
        "TinyCMA",
        "QODE",
        "MetaModelOnePlusOne",
        "LhsDE",
        "TinyLhsDE",
        "TinyQODE",
        "ChainMetaModelSQP",
        "MicroCMA",
        "MultiScaleCMA",
    ]
    optims = ["QODE"]
    optims = ["CMA", "LargeCMA", "OldCMA", "DE", "PSO", "Powell", "Cobyla", "SQP"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    dims = [2, 3, 5, 10, 20]
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, translation_factor=tf)
        for name in ["cigar", "sphere", "rastrigin"]
        for rotation in [True]
        for expo in [1.0, 5.0]
        for tf in [0.01, 0.1, 1.0, 10.0]
        for d in dims
    ]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def zp_ms_bbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on exponentiated problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """

    seedg = create_seed_generator(seed)
    optims = [
        "TinyCMA",
        "QODE",
        "MetaModelOnePlusOne",
        "LhsDE",
        "TinyLhsDE",
        "TinyQODE",
        "ChainMetaModelSQP",
        "MicroCMA",
        "MultiScaleCMA",
    ]
    optims = ["QODE"]
    optims = ["CMA", "LargeCMA", "OldCMA", "DE", "PSO", "Powell", "Cobyla", "SQP"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    dims = [2, 3, 5, 10, 20]
    functions = [
        ArtificialFunction(
            name, block_dimension=d, rotation=rotation, expo=expo, translation_factor=tf, zero_pen=True
        )
        for name in ["cigar", "sphere", "rastrigin"]
        for rotation in [True]
        for expo in [1.0, 5.0]
        for tf in [0.01, 0.1, 1.0, 10.0]
        for d in dims
    ]
    optims = ["QODE", "PSO", "SQOPSO", "DE", "CMA"]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


def nozp_noms_bbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on exponentiated problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """

    seedg = create_seed_generator(seed)
    optims = [
        "TinyCMA",
        "QODE",
        "MetaModelOnePlusOne",
        "LhsDE",
        "TinyLhsDE",
        "TinyQODE",
        "ChainMetaModelSQP",
        "MicroCMA",
        "MultiScaleCMA",
    ]
    optims = ["QODE"]
    optims = ["CMA", "LargeCMA", "OldCMA", "DE", "PSO", "Powell", "Cobyla", "SQP"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    dims = [2, 3, 5, 10, 20]
    functions = [
        ArtificialFunction(
            name, block_dimension=d, rotation=rotation, expo=expo, translation_factor=tf, zero_pen=False
        )
        for name in ["cigar", "sphere", "rastrigin"]
        for rotation in [True]
        for expo in [1.0, 5.0]
        for tf in [1.0]
        for d in dims
    ]
    optims = ["QODE", "PSO", "SQOPSO", "DE", "CMA"]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def pbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on exponentiated problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """
    seedg = create_seed_generator(seed)
    optims = [
        "OldCMA",
        "CMAbounded",
        "CMAsmall",
        "CMAstd",
        "CMApara",
        "CMAtuning",
        "DiagonalCMA",
        "FCMA",
        "RescaledCMA",
        "ASCMADEthird",
        "MultiCMA",
        "TripleCMA",
        "PolyCMA",
        "MultiScaleCMA",
        "DE",
        "OnePointDE",
        "GeneticDE",
        "TwoPointsDE",
        "PSO",
        "NGOptRW",
        "NGOpt",
    ]
    optims = ["ChainMetaModelSQP", "MetaModelOnePlusOne", "MetaModelDE"]
    optims = ["LargeCMA", "TinyCMA", "OldCMA", "MicroCMA"]
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["QrDE", "QODE", "LhsDE", "NGOpt", "NGOptRW"]
    optims = ["TinyCMA", "QODE", "MetaModelOnePlusOne", "LhsDE", "TinyLhsDE", "TinyQODE"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    dims = [40, 20]
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo)
        for name in ["cigar", "sphere", "rastrigin", "hm", "deceptivemultimodal"]
        for rotation in [True]
        for expo in [1.0, 3.0, 5.0, 7.0, 9.0]
        for d in dims
    ]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def zp_pbbob(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on exponentiated problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """
    seedg = create_seed_generator(seed)
    optims = [
        "OldCMA",
        "CMAbounded",
        "CMAsmall",
        "CMAstd",
        "CMApara",
        "CMAtuning",
        "DiagonalCMA",
        "FCMA",
        "RescaledCMA",
        "ASCMADEthird",
        "MultiCMA",
        "TripleCMA",
        "PolyCMA",
        "MultiScaleCMA",
        "DE",
        "OnePointDE",
        "GeneticDE",
        "TwoPointsDE",
        "PSO",
        "NGOptRW",
        "NGOpt",
    ]
    optims = ["ChainMetaModelSQP", "MetaModelOnePlusOne", "MetaModelDE"]
    optims = ["LargeCMA", "TinyCMA", "OldCMA", "MicroCMA"]
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["QrDE", "QODE", "LhsDE", "NGOpt", "NGOptRW"]
    optims = ["TinyCMA", "QODE", "MetaModelOnePlusOne", "LhsDE", "TinyLhsDE", "TinyQODE"]
    optims = ["QOPSO", "QORealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    dims = [40, 20]
    functions = [
        ArtificialFunction(name, block_dimension=d, rotation=rotation, expo=expo, zero_pen=True)
        for name in ["cigar", "sphere", "rastrigin", "hm", "deceptivemultimodal"]
        for rotation in [True]
        for expo in [1.0, 3.0, 5.0, 7.0, 9.0]
        for d in dims
    ]
    optims = ["QODE", "PSO", "SQOPSO", "DE", "CMA"]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 200, 300, 400, 500, 600, 700, 800]:
                for nw in [1, 10, 50]:
                    yield Experiment(function, optim, budget=budget, num_workers=nw, seed=next(seedg))


@registry.register
def illcondi(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on ill cond problems.
    Cigar, Ellipsoid.
    Both rotated and unrotated.
    Budget 100, 1000, 10000.
    Dimension 50.
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    optims = refactor_optims(optims)
    for optim in optims:
        for function in functions:
            for budget in [100, 1000, 10000]:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def illcondipara(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Testing optimizers on ill-conditionned parallel optimization.
    50 workers in parallel.
    """
    seedg = create_seed_generator(seed)
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    optims = get_optimizers("competitive", seed=next(seedg))
    optims = refactor_optims(optims)
    for function in functions:
        for budget in [100, 1000, 10000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=50, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def constrained_illconditioned_parallel(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Many optimizers on ill cond problems with constraints."""
    seedg = create_seed_generator(seed)
    functions = [
        ArtificialFunction(name, block_dimension=50, rotation=rotation)
        for name in ["cigar", "ellipsoid"]
        for rotation in [True, False]
    ]
    for func in functions:
        func.parametrization.register_cheap_constraint(_Constraint("sum", as_bool=False))
    optims = ["DE", "CMA", "NGOpt"]
    optims = refactor_optims(optims)  # type: ignore
    for function in functions:
        for budget in [400, 4000, 40000]:
            optims: tp.List[str] = get_optimizers("large", seed=next(seedg))  # type: ignore
            for optim in optims:
                yield Experiment(function, optim, budget=budget, num_workers=1, seed=next(seedg))


@registry.register
def ranknoisy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    Cigar, Altcigar, Ellipsoid, Altellipsoid.
    Dimension 200, 2000, 20000.
    Budget 25000, 50000, 100000.
    No rotation.
    Noise level 10.
    With or without noise dissymmetry.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [  # type: ignore
        "OptimisticNoisyOnePlusOne",
        "OptimisticDiscreteOnePlusOne",
        "NGOpt10",
    ]

    # optims += ["NGO", "Shiwa", "DiagonalCMA"] + sorted(
    #    x for x, y in ng.optimizers.registry.items() if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    # )
    optims = ["SPSA", "TinySPSA", "TBPSA", "NoisyOnePlusOne", "NoisyDiscreteOnePlusOne"]
    optims = get_optimizers("basics", "noisy", "splitters", "progressive", seed=next(seedg))  # type: ignore
    optims = refactor_optims(optims)
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [20000, 200, 2000]:
                for name in ["cigar", "altcigar", "ellipsoid", "altellipsoid"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(
                            name=name,
                            rotation=False,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def noisy(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Noisy optimization methods on a few noisy problems.
    Sphere, Rosenbrock, Cigar, Hm (= highly multimodal).
    Noise level 10.
    Noise dyssymmetry or not.
    Dimension 2, 20, 200, 2000.
    Budget 25000, 50000, 100000.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("progressive", seed=next(seedg)) + [  # type: ignore
        "OptimisticNoisyOnePlusOne",
        "OptimisticDiscreteOnePlusOne",
    ]
    optims += ["NGOpt10", "Shiwa", "DiagonalCMA"] + sorted(
        x
        for x, y in ng.optimizers.registry.items()
        if ("SPSA" in x or "TBPSA" in x or "ois" in x or "epea" in x or "Random" in x)
    )

    optims = refactor_optims(optims)
    for budget in [25000, 50000, 100000]:
        for optim in optims:
            for d in [2, 20, 200, 2000]:
                for name in ["sphere", "rosenbrock", "cigar", "hm"]:
                    for noise_dissymmetry in [False, True]:
                        function = ArtificialFunction(
                            name=name,
                            rotation=True,
                            block_dimension=d,
                            noise_level=10,
                            noise_dissymmetry=noise_dissymmetry,
                            translation_factor=1.0,
                        )
                        yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def paraalldes(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All DE methods on various functions. Parallel version.
    Dimension 5, 20, 100, 500, 2500.
    Sphere, Cigar, Hm, Ellipsoid.
    No rotation.
    """
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x and "Tune" in x):
            for rotation in [False]:
                for d in [5, 20, 100, 500, 2500]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(
                                function,
                                optim,
                                budget=budget,
                                seed=next(seedg),
                                num_workers=max(d, budget // 6),
                            )


@registry.register
def parahdbo4d(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All Bayesian optimization methods on various functions. Parallel version
    Dimension 20 and 2000.
    Budget 25, 31, 37, 43, 50, 60.
    Sphere, Cigar, Hm, Ellipsoid.
    No rotation.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(
            sorted(x for x, y in ng.optimizers.registry.items() if "BO" in x and "Tune" in x)
        ):
            for rotation in [False]:
                for d in [20, 2000]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(
                                function,
                                optim,
                                budget=budget,
                                seed=next(seedg),
                                num_workers=max(d, budget // 6),
                            )


@registry.register
def alldes(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All DE methods on various functions.
    Dimension 5, 20, 100.
    Sphere, Cigar, Hm, Ellipsoid.
    Budget 10, 100, 1000, 10000, 100000.
    """
    seedg = create_seed_generator(seed)
    for budget in [10, 100, 1000, 10000, 100000]:
        for optim in refactor_optims(
            sorted(x for x, y in ng.optimizers.registry.items() if "DE" in x or "Shiwa" in x)
        ):
            for rotation in [False]:
                for d in [5, 20, 100]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def hdbo4d(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """All Bayesian optimization methods on various functions.
    Budget 25, 31, 37, 43, 50, 60.
    Dimension 20.
    Sphere, Cigar, Hm, Ellipsoid.
    """
    seedg = create_seed_generator(seed)
    for budget in [25, 31, 37, 43, 50, 60]:
        for optim in refactor_optims(get_optimizers("all_bo", seed=next(seedg))):
            for rotation in [False]:
                for d in [20]:
                    for name in ["sphere", "cigar", "hm", "ellipsoid"]:
                        for u in [0]:
                            function = ArtificialFunction(
                                name=name,
                                rotation=rotation,
                                block_dimension=d,
                                useless_variables=d * u,
                                translation_factor=1.0,
                            )
                            yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def spsa_benchmark(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Some optimizers on a noisy optimization problem. This benchmark is based on the noisy benchmark.
    Budget 500, 1000, 2000, 4000, ... doubling... 128000.
    Rotation or not.
    Sphere, Sphere4, Cigar.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("spsa", seed=next(seedg))  # type: ignore
    optims += ["CMA", "OnePlusOne", "DE", "PSO"]
    optims = ["SQP", "NoisyDiscreteOnePlusOne", "NoisyBandit"]
    optims = ["NGOpt", "NGOptRW"]
    optims = refactor_optims(optims)
    for budget in [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        for optim in optims:
            for rotation in [True, False]:
                for name in ["sphere", "sphere4", "cigar"]:
                    function = ArtificialFunction(
                        name=name, rotation=rotation, block_dimension=20, noise_level=10
                    )
                    yield Experiment(function, optim, budget=budget, seed=next(seedg))


@registry.register
def realworld(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Realworld optimization. This experiment contains:

     - a subset of MLDA (excluding the perceptron: 10 functions rescaled or not.
     - ARCoating https://arxiv.org/abs/1904.02907: 1 function.
     - The 007 game: 1 function, noisy.
     - PowerSystem: a power system simulation problem.
     - STSP: a simple TSP problem.
     -  MLDA, except the Perceptron.

    Budget 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800.
    Sequential or 10-parallel or 100-parallel.
    """
    funcs: tp.List[tp.Union[ExperimentFunction, rl.agents.TorchAgentFunction]] = [
        _mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10)]
        for rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        # _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]

    # Adding ARCoating.
    funcs += [ARCoating()]
    funcs += [PowerSystem(), PowerSystem(13)]
    funcs += [STSP(), STSP(500)]
    funcs += [game.Game("war")]
    funcs += [game.Game("batawaf")]
    funcs += [game.Game("flip")]
    funcs += [game.Game("guesswho")]
    funcs += [game.Game("bigguesswho")]

    # 007 with 100 repetitions, both mono and multi architectures.
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {
        a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False)
        for a, m in modules.items()
    }
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    runner = rl.EnvironmentRunner(env.copy(), num_repetitions=100, max_step=50)
    for archi in ["mono", "multi"]:
        func = rl.agents.TorchAgentFunction(agents[archi], runner, reward_postprocessing=lambda x: 1 - x)
        funcs += [func]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def aquacrop_fao(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """FAO Crop simulator. Maximize yield."""

    funcs = [NgAquacrop(i, 300.0 + 150.0 * np.cos(i)) for i in range(3, 7)]
    # funcs = list(np.random.choice(funcs, 2))
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["NGOpt"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def fishing(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Lotka-Volterra equations"""
    funcs = [OptimizeFish(i) for i in [17, 35, 52, 70, 88, 105]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims += ["NGOpt", "NGOptRW", "ChainMetaModelSQP"]
    optims = ["NGOpt"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, seed=next(seedg))
                xp.function.parametrization.real_world = True
                if not xp.is_incoherent:
                    yield xp


@registry.register
def rocket(seed: tp.Optional[int] = None, seq: bool = False) -> tp.Iterator[Experiment]:
    """Rocket simulator. Maximize max altitude by choosing the thrust schedule, given a total thrust.
    Budget 25, 50, ..., 1600.
    Sequential or 30 workers."""
    funcs = [Rocket(i) for i in range(17)]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims += ["NGOpt", "NGOptRW", "ChainMetaModelSQP"]
    optims = ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["CMA", "PSO", "QODE", "QRDE", "MetaModelPSO"]
    if seq:
        optims += ["RBFGS", "LBFGSB", "MemeticDE"]
    optims = ["NGOpt"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["SQOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = [
        "NGOpt",
        "QOPSO",
        "SOPSO",
        "QODE",
        "SODE",
        "CMA",
        "DiagonalCMA",
        "MetaModelOnePlusOne",
        "MetaModelDE",
    ]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600]:
        for num_workers in [1] if seq else [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:  # list(np.random.choice(funcs, 3)):
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        skip_ci(reason="Too slow")
                        if not xp.is_incoherent:  # and np.random.choice([True, False, False]):
                            yield xp


@registry.register
def mono_rocket(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Sequential counterpart of the rocket problem."""
    return rocket(seed, seq=True)


@registry.register
def mixsimulator(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MixSimulator of power plants
    Budget 20, 40, ..., 1600.
    Sequential or 30 workers."""
    funcs = [OptimizeMix()]
    seedg = create_seed_generator(seed)
    optims: tp.List[str] = get_optimizers("basics", seed=next(seedg))  # type: ignore

    optims = refactor_optims(optims)
    for budget in [20, 40, 80, 160]:
        for num_workers in [1, 30]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def control_problem(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MuJoCo testbed. Learn linear policy for different control problems.
    Budget 500, 1000, 3000, 5000."""
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [
        Env(num_rollouts=num_rollouts, random_state=seed)
        for Env in [
            control.Swimmer,
            control.HalfCheetah,
            control.Hopper,
            control.Walker2d,
            control.Ant,
            control.Humanoid,
        ]
    ]

    sigmas = [0.1, 0.1, 0.1, 0.1, 0.01, 0.001]
    funcs2 = []
    for sigma, func in zip(sigmas, funcs):
        f = func.copy()
        param: ng.p.Tuple = f.parametrization.copy()  # type: ignore
        for array in param:
            array.set_mutation(sigma=sigma)  # type: ignore
        param.set_name(f"sigma={sigma}")

        f.parametrization = param
        f.parametrization.freeze()
        funcs2.append(f)
    optims = get_optimizers("basics")
    optims = ["NGOpt", "PSO", "CMA"]
    optims = refactor_optims(optims)
    for budget in [50, 75, 100, 150, 200, 250, 300, 400, 500, 1000, 3000, 5000, 8000, 16000, 32000, 64000]:
        for algo in optims:
            for fu in funcs2:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def neuro_control_problem(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MuJoCo testbed. Learn neural policies."""
    seedg = create_seed_generator(seed)
    num_rollouts = 1
    funcs = [
        Env(num_rollouts=num_rollouts, intermediate_layer_dim=(50,), random_state=seed)
        for Env in [
            control.Swimmer,
            control.HalfCheetah,
            control.Hopper,
            control.Walker2d,
            control.Ant,
            control.Humanoid,
        ]
    ]

    optims = ["CMA", "NGOpt4", "DiagonalCMA", "NGOpt8", "MetaModel", "ChainCMAPowell"]
    optims = ["NGOpt", "CMA", "PSO"]
    optims = refactor_optims(optims)
    for budget in [50, 500, 5000, 10000, 20000, 35000, 50000, 100000, 200000]:
        for algo in optims:
            for fu in funcs:
                xp = Experiment(fu, algo, budget, num_workers=1, seed=next(seedg))
                xp.function.parametrization.real_world = True
                xp.function.parametrization.neural = True
                if not xp.is_incoherent:
                    yield xp


@registry.register
def olympus_surfaces(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Olympus surfaces"""
    from nevergrad.functions.olympussurfaces import OlympusSurface

    funcs = []
    for kind in OlympusSurface.SURFACE_KINDS:
        for k in range(2, 5):
            for noise in ["GaussianNoise", "UniformNoise", "GammaNoise"]:
                for noise_scale in [0.5, 1]:
                    funcs.append(OlympusSurface(kind, 10**k, noise, noise_scale))

    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", seed=next(seedg))
    optims = ["NGOpt", "CMA"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def olympus_emulators(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Olympus emulators"""
    from nevergrad.functions.olympussurfaces import OlympusEmulator

    funcs = []
    for dataset_kind in OlympusEmulator.DATASETS:
        for model_kind in ["BayesNeuralNet", "NeuralNet"]:
            funcs.append(OlympusEmulator(dataset_kind, model_kind))

    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", "noisy", seed=next(seedg))
    optims = ["NGOpt", "CMA"]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def topology_optimization(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = ["CMA", "GeneticDE", "TwoPointsDE", "VoronoiDE", "DE", "PSO", "RandomSearch", "OnePlusOne"]
    optims = ["NGOpt"]
    optims = refactor_optims(optims)
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))


@registry.register
def sequential_topology_optimization(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    funcs = [TO(i) for i in [10, 20, 30, 40]]
    optims = ["CMA", "GeneticDE", "TwoPointsDE", "VoronoiDE", "DE", "PSO", "RandomSearch", "OnePlusOne"]
    optims = ["NGOpt"]
    optims = refactor_optims(optims)
    for budget in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960]:
        for optim in optims:
            for f in funcs:
                for nw in [1, 30]:
                    yield Experiment(f, optim, budget, num_workers=nw, seed=next(seedg))


@registry.register
def simple_tsp(seed: tp.Optional[int] = None, complex_tsp: bool = False) -> tp.Iterator[Experiment]:
    """Simple TSP problems. Please note that the methods we use could be applied or complex variants, whereas
    specialized methods can not always do it; therefore this comparisons from a black-box point of view makes sense
    even if white-box methods are not included though they could do this more efficiently.
    10, 100, 1000, 10000 cities.
    Budgets doubling from 25, 50, 100, 200, ... up  to 25600

    """
    funcs = [STSP(10**k, complex_tsp) for k in range(2, 6)]
    seedg = create_seed_generator(seed)
    optims = [
        "RotatedTwoPointsDE",
        "DiscreteLenglerOnePlusOne",
        "DiscreteDoerrOnePlusOne",
        "DiscreteBSOOnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "GeneticDE",
        "DE",
        "TwoPointsDE",
        "DiscreteOnePlusOne",
        "CMA",
        "MetaModel",
        "DiagonalCMA",
    ]
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]:
        for num_workers in [1]:  # , 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def complex_tsp(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of simple_tsp with non-planar term."""
    return simple_tsp(seed, complex_tsp=True)


@registry.register
def sequential_fastgames(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of policies for games, i.e. direct policy search.
    Budget 12800, 25600, 51200, 102400.
    Games: War, Batawaf, Flip, GuessWho,  BigGuessWho."""
    funcs = [game.Game(name) for name in ["war", "batawaf", "flip", "guesswho", "bigguesswho"]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("noisy", "splitters", "progressive", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [12800, 25600, 51200, 102400]:
        for num_workers in [1]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def powersystems(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Unit commitment problem, i.e. management of dams for hydroelectric planning."""
    funcs: tp.List[ExperimentFunction] = []
    for dams in [3, 5, 9, 13]:
        funcs += [PowerSystem(dams, depth=2, width=3)]
    seedg = create_seed_generator(seed)
    budgets = [3200, 6400, 12800]
    optims = get_optimizers("basics", "noisy", "splitters", "progressive", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in budgets:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for fu in funcs:
                        xp = Experiment(fu, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mlda(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed."""
    funcs: tp.List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10)]
        for rescale in [True, False]
    ]
    funcs += [
        _mlda.SammonMapping.from_mlda("Virus", rescale=False),
        _mlda.SammonMapping.from_mlda("Virus", rescale=True),
        # _mlda.SammonMapping.from_mlda("Employees"),
    ]
    funcs += [_mlda.Perceptron.from_mlda(name) for name in ["quadratic", "sine", "abs", "heaviside"]]
    funcs += [_mlda.Landscape(transform) for transform in [None, "square", "gaussian"]]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("basics", seed=next(seedg))
    optims = refactor_optims(optims)
    for budget in [25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        xp.function.parametrization.real_world = True
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def mldakmeans(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """MLDA (machine learning and data analysis) testbed, restricted to the K-means part."""
    funcs: tp.List[ExperimentFunction] = [
        _mlda.Clustering.from_mlda(name, num, rescale)
        for name, num in [("Ruspini", 5), ("German towns", 10), ("Ruspini", 50), ("German towns", 100)]
        for rescale in [True, False]
    ]
    seedg = create_seed_generator(seed)
    optims = get_optimizers("splitters", "progressive", seed=next(seedg))
    optims += ["DE", "CMA", "PSO", "TwoPointsDE", "RandomSearch"]
    optims = ["QODE", "QRDE"]
    optims = ["NGOpt"]
    optims = refactor_optims(optims)
    for budget in [1000, 10000]:
        for num_workers in [1, 10, 100]:
            if num_workers < budget:
                for algo in optims:
                    for func in funcs:
                        xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                        if not xp.is_incoherent:
                            yield xp


@registry.register
def image_similarity(
    seed: tp.Optional[int] = None, with_pgan: bool = False, similarity: bool = True
) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE == similarity
    ]
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for func in funcs:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                skip_ci(reason="too slow")
                if not xp.is_incoherent:
                    yield xp


@registry.register
def image_similarity_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity, using PGan as a representation."""
    return image_similarity(seed, with_pgan=True)


@registry.register
def image_single_quality(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity, but based on image quality assessment."""
    return image_similarity(seed, with_pgan=False, similarity=False)


@registry.register
def image_single_quality_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_pgan, but based on image quality assessment."""
    return image_similarity(seed, with_pgan=True, similarity=False)


@registry.register
def image_multi_similarity(
    seed: tp.Optional[int] = None, cross_valid: bool = False, with_pgan: bool = False
) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structured_moo", seed=next(seedg))
    funcs: tp.List[ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE
    ]
    base_values: tp.List[tp.Any] = [func(func.parametrization.sample().value) for func in funcs]
    if cross_valid:
        skip_ci(reason="Too slow")
        mofuncs: tp.List[tp.Any] = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
            funcs, pareto_size=25
        )
    else:
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=base_values)]
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def image_multi_similarity_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity, using PGan as a representation."""
    return image_multi_similarity(seed, with_pgan=True)


@registry.register
def image_multi_similarity_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_multi_similarity with cross-validation."""
    return image_multi_similarity(seed, cross_valid=True)


@registry.register
def image_multi_similarity_pgan_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_multi_similarity with cross-validation."""
    return image_multi_similarity(seed, cross_valid=True, with_pgan=True)


@registry.register
def image_quality_proxy(seed: tp.Optional[int] = None, with_pgan: bool = False) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    iqa, blur, brisque = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in (imagesxp.imagelosses.Koncept512, imagesxp.imagelosses.Blur, imagesxp.imagelosses.Brisque)
    ]
    # TODO: add the proxy info in the parametrization.
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for algo in optims:
            for func in [blur, brisque]:
                # We optimize on blur or brisque and check performance on iqa.
                sfunc = helpers.SpecialEvaluationExperiment(func, evaluation=iqa)
                sfunc.add_descriptors(non_proxy_function=False)
                xp = Experiment(sfunc, algo, budget, num_workers=1, seed=next(seedg))
                yield xp


@registry.register
def image_quality_proxy_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return image_quality_proxy(seed, with_pgan=True)


@registry.register
def image_quality(
    seed: tp.Optional[int] = None, cross_val: bool = False, with_pgan: bool = False, num_images: int = 1
) -> tp.Iterator[Experiment]:
    """Optimizing images for quality:
    we optimize K512, Blur and Brisque.

    With num_images > 1, we are doing morphing.
    """
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))
    # We optimize func_blur or func_brisque and check performance on func_iqa.
    funcs: tp.List[ExperimentFunction] = [
        imagesxp.Image(loss=loss, with_pgan=with_pgan, num_images=num_images)
        for loss in (
            imagesxp.imagelosses.Koncept512,
            imagesxp.imagelosses.Blur,
            imagesxp.imagelosses.Brisque,
        )
    ]
    # TODO: add the proxy info in the parametrization.
    mofuncs: tp.Sequence[ExperimentFunction]
    if cross_val:
        mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
            experiments=[funcs[0], funcs[2]],
            # Blur is not good enough as an IQA for being in the list.
            training_only_experiments=[funcs[1]],
            pareto_size=16,
        )
    else:
        upper_bounds = [func(func.parametrization.value) for func in funcs]
        mofuncs = [fbase.MultiExperiment(funcs, upper_bounds=upper_bounds)]  # type: ignore
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                for func in mofuncs:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


@registry.register
def morphing_pgan_quality(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return image_quality(seed, with_pgan=True, num_images=2)


@registry.register
def image_quality_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, cross_val=True)


@registry.register
def image_quality_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, with_pgan=True)


@registry.register
def image_quality_cv_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_quality with cross-validation."""
    return image_quality(seed, cross_val=True, with_pgan=True)


@registry.register
def image_similarity_and_quality(
    seed: tp.Optional[int] = None, cross_val: bool = False, with_pgan: bool = False
) -> tp.Iterator[Experiment]:
    """Optimizing images: artificial criterion for now."""
    seedg = create_seed_generator(seed)
    optims: tp.List[tp.Any] = get_optimizers("structured_moo", seed=next(seedg))

    # 3 losses functions including 2 iqas.
    func_iqa = imagesxp.Image(loss=imagesxp.imagelosses.Koncept512, with_pgan=with_pgan)
    func_blur = imagesxp.Image(loss=imagesxp.imagelosses.Blur, with_pgan=with_pgan)
    base_blur_value: float = func_blur(func_blur.parametrization.value)  # type: ignore
    optims = refactor_optims(optims)
    for func in [
        imagesxp.Image(loss=loss, with_pgan=with_pgan)
        for loss in imagesxp.imagelosses.registry.values()
        if loss.REQUIRES_REFERENCE
    ]:

        # Creating a reference value.
        base_value: float = func(func.parametrization.value)  # type: ignore
        mofuncs: tp.Iterable[fbase.ExperimentFunction]
        if cross_val:
            mofuncs = helpers.SpecialEvaluationExperiment.create_crossvalidation_experiments(
                training_only_experiments=[func, func_blur], experiments=[func_iqa], pareto_size=16
            )
        else:
            mofuncs = [
                fbase.MultiExperiment(
                    [func, func_blur, func_iqa], upper_bounds=[base_value, base_blur_value, 100.0]
                )
            ]
        for budget in [100 * 5**k for k in range(3)]:
            for algo in optims:
                for mofunc in mofuncs:
                    xp = Experiment(mofunc, algo, budget, num_workers=1, seed=next(seedg))
                    yield xp


@registry.register
def image_similarity_and_quality_cv(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_and_quality with cross-validation."""
    return image_similarity_and_quality(seed, cross_val=True)


@registry.register
def image_similarity_and_quality_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_and_quality with cross-validation."""
    return image_similarity_and_quality(seed, with_pgan=True)


@registry.register
def image_similarity_and_quality_cv_pgan(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of image_similarity_and_quality with cross-validation."""
    return image_similarity_and_quality(seed, cross_val=True, with_pgan=True)


@registry.register
def double_o_seven(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Optimization of policies for the 007 game.
    Sequential or 10-parallel or 100-parallel. Various numbers of averagings: 1, 10 or 100."""
    # pylint: disable=too-many-locals
    seedg = create_seed_generator(seed)
    base_env = rl.envs.DoubleOSeven(verbose=False)
    random_agent = rl.agents.Agent007(base_env)
    modules = {"mono": rl.agents.Perceptron, "multi": rl.agents.DenseNet}
    agents = {
        a: rl.agents.TorchAgent.from_module_maker(base_env, m, deterministic=False)
        for a, m in modules.items()
    }
    env = base_env.with_agent(player_0=random_agent).as_single_agent()
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    optimizers: tp.List[tp.Any] = [
        "PSO",
        dde,
        "MetaTuneRecentering",
        "DiagonalCMA",
        "TBPSA",
        "SPSA",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "MetaModelPSO",
    ]
    optimizers = ["NGOpt", "NGOptRW"]
    optimizers = refactor_optims(optimizers)  # type: ignore
    for num_repetitions in [1, 10, 100]:
        for archi in ["mono", "multi"]:
            for optim in optimizers:
                for env_budget in [5000, 10000, 20000, 40000]:
                    for num_workers in [1, 10, 100]:
                        # careful, not threadsafe
                        runner = rl.EnvironmentRunner(
                            env.copy(), num_repetitions=num_repetitions, max_step=50
                        )
                        func = rl.agents.TorchAgentFunction(
                            agents[archi], runner, reward_postprocessing=lambda x: 1 - x
                        )
                        opt_budget = env_budget // num_repetitions
                        xp = Experiment(
                            func,
                            optim,
                            budget=opt_budget,
                            num_workers=num_workers,
                            seed=next(seedg),
                        )
                        xp.function.parametrization.real_world = True
                        yield xp


@registry.register
def multiobjective_example(
    seed: tp.Optional[int] = None, hd: bool = False, many: bool = False
) -> tp.Iterator[Experiment]:
    """Optimization of 2 and 3 objective functions in Sphere, Ellipsoid, Cigar, Hm.
    Dimension 6 and 7.
    Budget 100 to 3200
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structure", "structured_moo", seed=next(seedg))
    optims += [
        ng.families.DifferentialEvolution(multiobjective_adaptation=False).set_name("DE-noadapt"),
        ng.families.DifferentialEvolution(crossover="twopoints", multiobjective_adaptation=False).set_name(
            "TwoPointsDE-noadapt"
        ),
    ]
    optims += ["DiscreteOnePlusOne", "DiscreteLenglerOnePlusOne"]
    optims = ["PymooNSGA2", "PymooBatchNSGA2", "LPCMA", "VLPCMA", "CMA"]
    optims = ["LPCMA", "VLPCMA", "CMA"]
    popsizes = [20, 40, 80]
    optims += [
        ng.families.EvolutionStrategy(
            recombination_ratio=recomb, only_offsprings=only, popsize=pop, offsprings=pop * 5
        )
        for only in [True, False]
        for recomb in [0.1, 0.5]
        for pop in popsizes
    ]

    optims = refactor_optims(optims)
    mofuncs: tp.List[fbase.MultiExperiment] = []
    dim = 2000 if hd else 7
    for name1, name2 in itertools.product(["sphere"], ["sphere", "hm"]):
        mofuncs.append(
            fbase.MultiExperiment(
                [
                    ArtificialFunction(name1, block_dimension=dim),
                    ArtificialFunction(name2, block_dimension=dim),
                ]
                + (
                    [
                        # Addendum for many-objective optim.
                        ArtificialFunction(name1, block_dimension=dim),
                        ArtificialFunction(name2, block_dimension=dim),
                    ]
                    if many
                    else []
                ),
                upper_bounds=[100, 100] * (2 if many else 1),
            )
        )
        mofuncs.append(
            fbase.MultiExperiment(
                [
                    ArtificialFunction(name1, block_dimension=dim - 1),
                    ArtificialFunction("sphere", block_dimension=dim - 1),
                    ArtificialFunction(name2, block_dimension=dim - 1),
                ]
                + (
                    [
                        ArtificialFunction(
                            name1, block_dimension=dim - 1
                        ),  # Addendum for many-objective optim.
                        ArtificialFunction("sphere", block_dimension=dim - 1),
                        ArtificialFunction(name2, block_dimension=dim - 1),
                    ]
                    if many
                    else []
                ),
                upper_bounds=[100, 100, 100.0] * (2 if many else 1),
            )
        )
    for mofunc in mofuncs:
        for optim in optims:
            for budget in [100, 200, 400, 800, 1600, 3200]:
                for nw in [1, 100]:
                    xp = Experiment(mofunc, optim, budget=budget, num_workers=nw, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def multiobjective_example_hd(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of moo with high dimension."""
    return multiobjective_example(seed, hd=True)


@registry.register
def multiobjective_example_many_hd(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of moo with high dimension and more objective functions."""
    return multiobjective_example(seed, hd=True, many=True)


@registry.register
def multiobjective_example_many(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of moo with more objective functions."""
    return multiobjective_example(seed, many=True)


@registry.register
def pbt(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optimizers = [
        "CMA",
        "TwoPointsDE",
        "Shiwa",
        "OnePlusOne",
        "DE",
        "PSO",
        "NaiveTBPSA",
        "RecombiningOptimisticNoisyDiscreteOnePlusOne",
        "PortfolioNoisyDiscreteOnePlusOne",
    ]  # type: ignore
    optimizers = refactor_optims(optimizers)
    for func in PBT.itercases():
        for optim in optimizers:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def far_optimum_es(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    # prepare list of parameters to sweep for independent variables
    seedg = create_seed_generator(seed)
    optims = get_optimizers("es", "basics", seed=next(seedg))  # type: ignore
    optims = refactor_optims(optims)
    for func in FarOptimumFunction.itercases():
        for optim in optims:
            for budget in [100, 400, 1000, 4000, 10000]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def ceviche(
    seed: tp.Optional[int] = None,
) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    instrum = ng.p.Array(shape=(40, 40), lower=0.0, upper=1.0).set_integer_casting()
    func = ExperimentFunction(photonics_ceviche, instrum.set_name("transition"))
    algos = [
        "DiagonalCMA",
        "PSO",
        "DE",
        "CMA",
        "OnePlusOne",
        "LognormalDiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "MetaModel",
        "MetaModelDE",
        "MetaModelDSproba",
        "MetaModelOnePlusOne",
        "MetaModelPSO",
        "MetaModelQODE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModel",
        "NeuralMetaModelDE",
        "NeuralMetaModelTwoPointsDE",
        "RFMetaModel",
        "RFMetaModelDE",
        "RFMetaModelOnePlusOne",
        "RFMetaModelPSO",
        "RFMetaModelTwoPointsDE",
        "SVMMetaModel",
        "SVMMetaModelDE",
        "SVMMetaModelPSO",
        "SVMMetaModelTwoPointsDE",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "SmoothDiscreteLognormalOnePlusOne",
        "SmoothLognormalDiscreteOnePlusOne",
        "UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne",
        "SuperSmoothRecombiningDiscreteLognormalOnePlusOne",
        "SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne",
        "RecombiningDiscreteLognormalOnePlusOne",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "UltraSmoothDiscreteLognormalOnePlusOne",
        "ZetaSmoothDiscreteLognormalOnePlusOne",
        "SuperSmoothDiscreteLognormalOnePlusOne",
    ]
    algos = [a for a in algos if a in list(ng.optimizers.registry.keys())]
    # print(algos)
    algo = np.random.choice(algos)
    print(algo)
    for optim in [algo]:
        for budget in [20, 50, 100, 160, 240]:
            yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def multi_ceviche(
    seed: tp.Optional[int] = None,
    c0: bool = False,
) -> tp.Iterator[Experiment]:
    seedg = create_seed_generator(seed)
    algos = [
        "DiagonalCMA",
        "PSO",
        "DE",
        "CMA",
        "OnePlusOne",
        "LognormalDiscreteOnePlusOne",
        "DiscreteLenglerOnePlusOne",
        "MetaModel",
        "MetaModelDE",
        "MetaModelDSproba",
        "MetaModelOnePlusOne",
        "MetaModelPSO",
        "MetaModelQODE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModel",
        "NeuralMetaModelDE",
        "NeuralMetaModelTwoPointsDE",
        "RFMetaModel",
        "RFMetaModelDE",
        "RFMetaModelOnePlusOne",
        "RFMetaModelPSO",
        "RFMetaModelTwoPointsDE",
        "SVMMetaModel",
        "SVMMetaModelDE",
        "SVMMetaModelPSO",
        "SVMMetaModelTwoPointsDE",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "SmoothDiscreteLognormalOnePlusOne",
        "SmoothLognormalDiscreteOnePlusOne",
        "UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne",
        "SuperSmoothRecombiningDiscreteLognormalOnePlusOne",
        "SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne",
        "RecombiningDiscreteLognormalOnePlusOne",
        "RandRecombiningDiscreteLognormalOnePlusOne",
        "UltraSmoothDiscreteLognormalOnePlusOne",
        "ZetaSmoothDiscreteLognormalOnePlusOne",
        "SuperSmoothDiscreteLognormalOnePlusOne",
    ]
    algos = [a for a in algos if a in list(ng.optimizers.registry.keys())]
    # print(algos)
    algo = np.random.choice(algos)
    print(algo)
    for benchmark_type in [np.random.randint(4)]:
        shape = tuple([int(p) for p in list(photonics_ceviche(None, benchmark_type))])  # type: ignore
        name = photonics_ceviche("name", benchmark_type) + str(shape)  # type: ignore
        print(f"Shape = {shape} {type(shape)} {type(shape[0])}")
        if c0:
            instrum = ng.p.Array(shape=shape, lower=0.0, upper=1.0)
        else:
            instrum = ng.p.Array(shape=shape, lower=0.0, upper=1.0).set_integer_casting()

        def pc(x):
            return photonics_ceviche(x, benchmark_type)

        instrum.set_name(name)
        func = ExperimentFunction(pc, instrum)
        # func.add_descriptor(name=name)
        # func.parametrization.set_name(name)
        print(f"name = {name}")
        for optim in [algo]:
            for budget in [20, 50, 90]:
                yield Experiment(func, optim, budget=budget, seed=next(seedg))


@registry.register
def multi_ceviche_c0(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of multi_ceviche with continuous permittivities."""
    return multi_ceviche(seed, c0=True)


@registry.register
def photonics(
    seed: tp.Optional[int] = None,
    as_tuple: bool = False,
    small: bool = False,
    ultrasmall: bool = False,
    verysmall: bool = False,
) -> tp.Iterator[Experiment]:
    """Too small for being interesting: Bragg mirror + Chirped + Morpho butterfly."""
    seedg = create_seed_generator(seed)
    divider = 2 if small else 1
    if ultrasmall or verysmall:
        divider = 4
    optims = get_optimizers("es", "basics", "splitters", seed=next(seedg))  # type: ignore
    optims = [
        "MemeticDE",
        "PSO",
        "DE",
        "CMA",
        "OnePlusOne",
        "TwoPointsDE",
        "GeneticDE",
        "ChainMetaModelSQP",
        "MetaModelDE",
        "SVMMetaModelDE",
        "RFMetaModelDE",
        "RBFGS",
        "LBFGSB",
    ]
    optims = ["QrDE", "QODE", "RFMetaModelDE"]
    optims = ["PCABO"]
    optims = ["PCABO", "NGOpt", "QODE"]
    optims = ["QOPSO"]  # , "QORealSpacePSO", "RealSpacePSO"]
    optims = ["MicroCMA", "MiniCMA", "QODE", "TinyDE", "MicroDE", "NGOpt"]
    optims = ["NGOpt"]
    optims = ["SQOPSO"]
    optims = refactor_optims(optims)
    for method in ["clipping", "tanh"]:  # , "arctan"]:
        for name in (
            ["bragg"]
            if ultrasmall
            else (
                ["cf_photosic_reference", "cf_photosic_realistic"]
                if verysmall
                else ["bragg", "chirped", "morpho", "cf_photosic_realistic", "cf_photosic_reference"]
            )
        ):
            func = Photonics(
                name,
                4 * ((60 // divider) // 4) if name == "morpho" else 80 // divider,
                bounding_method=method,
                as_tuple=as_tuple,
            )
            for budget in [1e1, 1e2, 1e3]:
                for algo in optims:
                    xp = Experiment(func, algo, int(budget), num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


@registry.register
def photonics2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=True)


@registry.register
def ultrasmall_photonics(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=False, small=True, ultrasmall=True)


@registry.register
def ultrasmall_photonics2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=True, small=True, ultrasmall=True)


@registry.register
def verysmall_photonics(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=False, small=True, verysmall=True)


@registry.register
def verysmall_photonics2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=True, small=True, verysmall=True)


@registry.register
def small_photonics(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=False, small=True)


@registry.register
def small_photonics2(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Counterpart of yabbob with higher dimensions."""
    return photonics(seed, as_tuple=True, small=True)


@registry.register
def adversarial_attack(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Pretrained ResNes50 under black-box attacked.
    Square attacks:
    100 queries ==> 0.1743119266055046
    200 queries ==> 0.09043250327653997
    300 queries ==> 0.05111402359108781
    400 queries ==> 0.04325032765399738
    1700 queries ==> 0.001310615989515072
    """
    seedg = create_seed_generator(seed)
    optims = get_optimizers("structure", "structured_moo", seed=next(seedg))
    folder = os.environ.get("NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER", None)
    # folder = "/datasets01/imagenet_full_size/061417/val"

    if folder is None:
        warnings.warn(
            "Using random images, set variable NEVERGRAD_ADVERSARIAL_EXPERIMENT_FOLDER to specify a folder"
        )
    optims = refactor_optims(optims)
    for func in imagesxp.ImageAdversarial.make_folder_functions(folder=folder):
        for budget in [100, 200, 300, 400, 1700]:
            for num_workers in [1]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                    yield xp


def pbo_suite(seed: tp.Optional[int] = None, reduced: bool = False) -> tp.Iterator[Experiment]:
    # Discrete, unordered.
    dde = ng.optimizers.DifferentialEvolution(crossover="dimension").set_name("DiscreteDE")
    seedg = create_seed_generator(seed)
    index = 0
    list_optims = [
        "DiscreteOnePlusOne",
        "Shiwa",
        "CMA",
        "PSO",
        "TwoPointsDE",
        "DE",
        "OnePlusOne",
        "AdaptiveDiscreteOnePlusOne",
        "CMandAS2",
        "PortfolioDiscreteOnePlusOne",
        "DoubleFastGADiscreteOnePlusOne",
        "MultiDiscrete",
        "cGA",
        dde,
    ]
    if reduced:
        list_optims = [
            x
            for x in ng.optimizers.registry.keys()
            if "iscre" in x and "ois" not in x and "ptim" not in x and "oerr" not in x
        ]
    list_optims = ["NGOpt", "NGOptRW"]
    list_optims = refactor_optims(list_optims)
    for dim in [16, 64, 100]:
        for fid in range(1, 24):
            for iid in range(1, 5):
                index += 1
                if reduced and index % 13:
                    continue
                for instrumentation in ["Softmax", "Ordered", "Unordered"]:
                    try:
                        func = iohprofiler.PBOFunction(fid, iid, dim, instrumentation=instrumentation)
                        func.add_descriptors(instrum_str=instrumentation)
                    except ModuleNotFoundError as e:
                        raise fbase.UnsupportedExperiment("IOHexperimenter needs to be installed") from e
                    for optim in list_optims:
                        for nw in [1, 10]:
                            for budget in [100, 1000, 10000]:
                                yield Experiment(func, optim, num_workers=nw, budget=budget, seed=next(seedg))  # type: ignore


@registry.register
def pbo_reduced_suite(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    return pbo_suite(seed, reduced=True)


def causal_similarity(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Finding the best causal graph"""
    # pylint: disable=import-outside-toplevel
    from nevergrad.functions.causaldiscovery import CausalDiscovery

    seedg = create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    func = CausalDiscovery()
    optims = refactor_optims(optims)
    for budget in [100 * 5**k for k in range(3)]:
        for num_workers in [1]:
            for algo in optims:
                xp = Experiment(func, algo, budget, num_workers=num_workers, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


def unit_commitment(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Unit commitment problem."""
    seedg = create_seed_generator(seed)
    optims = ["CMA", "NGOpt8", "DE", "PSO", "RecES", "RecMixES", "RecMutDE", "ParametrizationDE"]
    optims = refactor_optims(optims)
    for num_timepoint in [5, 10, 20]:
        for num_generator in [3, 8]:
            func = UnitCommitmentProblem(num_timepoints=num_timepoint, num_generators=num_generator)
            for budget in [100 * 5**k for k in range(3)]:
                for algo in optims:
                    xp = Experiment(func, algo, budget, num_workers=1, seed=next(seedg))
                    if not xp.is_incoherent:
                        yield xp


def team_cycling(seed: tp.Optional[int] = None) -> tp.Iterator[Experiment]:
    """Experiment to optimise team pursuit track cycling problem."""
    seedg = create_seed_generator(seed)
    optims = ["NGOpt10", "CMA", "DE"]
    funcs = [Cycling(num) for num in [30, 31, 61, 22, 23, 45]]
    optims = refactor_optims(optims)
    for function in funcs:
        for budget in [3000]:
            for optim in optims:
                xp = Experiment(function, optim, budget=budget, num_workers=10, seed=next(seedg))
                if not xp.is_incoherent:
                    yield xp


@registry.register
def lsgo() -> tp.Iterator[Experiment]:
    optims = [
        "Shiwa",
        "Cobyla",
        "Powell",
        "CMandAS2",
        "SQP",
        "DE",
        "TwoPointsDE",
        "CMA",
        "PSO",
        "OnePlusOne",
        "RBFGS",
    ]
    optims = ["PSO", "RealPSO"]
    optims = ["CMA", "PSO", "SQOPSO", "TinyCMA", "Cobyla"]
    optims = ["TwoPointsDE", "DE", "LhsDE"]
    optims = [
        "DE",
        "TwoPointsDE",
        "VoronoiDE",
        "RotatedTwoPointsDE",
        "LhsDE",
        "QrDE",
        "QODE",
        "SODE",
        "NoisyDE",
        "AlmostRotationInvariantDE",
        "RotationInvariantDE",
        "DiscreteDE",
        "RecMutDE",
        "MutDE",
        "OnePointDE",
        "ParametrizationDE",
        "MiniDE",
        "MiniLhsDE",
        "MiniQrDE",
        "BPRotationInvariantDE",
        "HSDE",
        "LhsHSDE",
        "TinyLhsDE",
        "TinyQODE",
        "MetaModelDE",
        "MetaModelQODE",
        "NeuralMetaModelDE",
        "SVMMetaModelDE",
        "RFMetaModelDE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModelTwoPointsDE",
        "SVMMetaModelTwoPointsDE",
        "RFMetaModelTwoPointsDE",
        "GeneticDE",
        "MemeticDE",
        "QNDE",
    ]
    optims = ["CMA", "NGOpt", "NGOptRW"]
    optims = ["DiagonalCMA", "TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = ["TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = refactor_optims(optims)
    for i in range(1, 16):  # [np.random.choice(list(range(1, 16)))]:
        for optim in optims:
            for budget in [120000, 600000, 3000000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)


@registry.register
def smallbudget_lsgo() -> tp.Iterator[Experiment]:
    optims = [
        "Shiwa",
        "Cobyla",
        "Powell",
        "CMandAS2",
        "SQP",
        "DE",
        "TwoPointsDE",
        "CMA",
        "PSO",
        "OnePlusOne",
        "RBFGS",
    ]
    optims = ["PSO", "RealPSO"]
    optims = ["CMA", "PSO", "SQOPSO", "TinyCMA", "Cobyla"]
    optims = ["TwoPointsDE", "DE", "LhsDE"]
    optims = [
        "DE",
        "TwoPointsDE",
        "VoronoiDE",
        "RotatedTwoPointsDE",
        "LhsDE",
        "QrDE",
        "QODE",
        "SODE",
        "NoisyDE",
        "AlmostRotationInvariantDE",
        "RotationInvariantDE",
        "DiscreteDE",
        "RecMutDE",
        "MutDE",
        "OnePointDE",
        "ParametrizationDE",
        "MiniDE",
        "MiniLhsDE",
        "MiniQrDE",
        "BPRotationInvariantDE",
        "HSDE",
        "LhsHSDE",
        "TinyLhsDE",
        "TinyQODE",
        "MetaModelDE",
        "MetaModelQODE",
        "NeuralMetaModelDE",
        "SVMMetaModelDE",
        "RFMetaModelDE",
        "MetaModelTwoPointsDE",
        "NeuralMetaModelTwoPointsDE",
        "SVMMetaModelTwoPointsDE",
        "RFMetaModelTwoPointsDE",
        "GeneticDE",
        "MemeticDE",
        "QNDE",
    ]
    optims = ["CMA", "NGOpt", "NGOptRW"]
    optims = ["DiagonalCMA", "TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = ["TinyQODE", "OpoDE", "OpoTinyDE"]
    optims = refactor_optims(optims)
    for i in range(1, 16):  # [np.random.choice(list(range(1, 16)))]:
        for optim in optims:
            for budget in [1200, 6000, 30000]:
                yield Experiment(lsgo_makefunction(i).instrumented(), optim, budget=budget)
