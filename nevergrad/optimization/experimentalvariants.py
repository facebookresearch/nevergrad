# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .oneshot import SamplingSearch
from .differentialevolution import DifferentialEvolution
from .optimizerlib import RandomSearchMaker, SQP, LHSSearch, DE, RandomSearch, MetaRecentering  # type: ignore
from .optimizerlib import (
    ParametrizedOnePlusOne,
    ParametrizedCMA,
    ConfiguredPSO,
    ConfSplitOptimizer,
    ParametrizedBO,
    EMNA,
)
from .optimizerlib import CMA, Chaining, PSO, BO

# DE
OnePointDE = DifferentialEvolution(crossover="onepoint").set_name(
    "OnePointDE", register=True
)
ParametrizationDE = DifferentialEvolution(crossover="parametrization").set_name(
    "ParametrizationDE", register=True
)
MiniDE = DifferentialEvolution(scale="mini").set_name("MiniDE", register=True)
MiniLhsDE = DifferentialEvolution(initialization="LHS", scale="mini").set_name(
    "MiniLhsDE", register=True
)
MiniQrDE = DifferentialEvolution(initialization="QR", scale="mini").set_name(
    "MiniQrDE", register=True
)
AlmostRotationInvariantDEAndBigPop = DifferentialEvolution(
    crossover=0.9, popsize="dimension"
).set_name("AlmostRotationInvariantDEAndBigPop", register=True)
BPRotationInvariantDE = DifferentialEvolution(crossover=1.0, popsize="large").set_name(
    "BPRotationInvariantDE", register=True
)

# CMA
MilliCMA = ParametrizedCMA(scale=1e-3).set_name("MilliCMA", register=True)
MicroCMA = ParametrizedCMA(scale=1e-6).set_name("MicroCMA", register=True)
FCMAs03 = ParametrizedCMA(fcmaes=True, scale=0.3).set_name("FCMAs03", register=True)
FCMAp13 = ParametrizedCMA(fcmaes=True, scale=0.1, popsize=13).set_name("FCMAp13", register=True)

# OnePlusOne
FastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="fastga").set_name(
    "FastGADiscreteOnePlusOne", register=True
)
DoubleFastGAOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="doublefastga"
).set_name("DoubleFastGAOptimisticNoisyDiscreteOnePlusOne", register=True)
FastGAOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="fastga"
).set_name("FastGAOptimisticNoisyDiscreteOnePlusOne", register=True)
FastGANoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="random", mutation="fastga"
).set_name("FastGANoisyDiscreteOnePlusOne", register=True)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="portfolio").set_name(
    "PortfolioDiscreteOnePlusOne", register=True
)
PortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="portfolio"
).set_name("PortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)
PortfolioNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="random", mutation="portfolio"
).set_name("PortfolioNoisyDiscreteOnePlusOne", register=True)
RecombiningOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="discrete", noise_handling="optimistic"
).set_name("RecombiningOptimisticNoisyDiscreteOnePlusOne", register=True)

# BO
RBO = ParametrizedBO(initialization="random").set_name("RBO", register=True)
QRBO = ParametrizedBO(initialization="Hammersley").set_name("QRBO", register=True)
MidQRBO = ParametrizedBO(initialization="Hammersley", middle_point=True).set_name(
    "MidQRBO", register=True
)
LBO = ParametrizedBO(initialization="LHS").set_name("LBO", register=True)

# PSO
WidePSO = ConfiguredPSO(transform="arctan", wide=True).set_name(
    "WidePSO", register=True
)  # non-standard init

# EMNA
IsoEMNA = EMNA(naive=False).set_name("IsoEMNA", register=True)
NaiveAnisoEMNA = EMNA(isotropic=False).set_name("NaiveAnisoEMNA", register=True)
AnisoEMNA = EMNA(naive=False, isotropic=False).set_name("AnisoEMNA", register=True)

# Recentering
MetaCauchyRecentering = SamplingSearch(
    cauchy=True, autorescale=True, sampler="Hammersley", scrambled=True
).set_name("MetaCauchyRecentering", register=True)
Recentering1ScrHammersleySearch = SamplingSearch(
    scale=0.1, sampler="Hammersley", scrambled=True
).set_name("Recentering1ScrHammersleySearch", register=True)
Recentering4ScrHammersleySearch = SamplingSearch(
    scale=0.4, sampler="Hammersley", scrambled=True
).set_name("Recentering4ScrHammersleySearch", register=True)
CauchyRecentering4ScrHammersleySearch = SamplingSearch(
    scale=0.4, cauchy=True, sampler="Hammersley", scrambled=True
).set_name("CauchyRecentering4ScrHammersleySearch", register=True)
Recentering1ScrHaltonSearch = SamplingSearch(
    scale=0.1, sampler="Halton", scrambled=True
).set_name("Recentering1ScrHaltonSearch", register=True)
Recentering4ScrHaltonSearch = SamplingSearch(
    scale=0.4, sampler="Halton", scrambled=True
).set_name("Recentering4ScrHaltonSearch", register=True)
Recentering7ScrHammersleySearch = SamplingSearch(
    scale=0.7, sampler="Hammersley", scrambled=True
).set_name("Recentering7ScrHammersleySearch", register=True)
CauchyRecentering7ScrHammersleySearch = SamplingSearch(
    scale=0.7, cauchy=True, sampler="Hammersley", scrambled=True
).set_name("CauchyRecentering7ScrHammersleySearch", register=True)
Recentering20ScrHaltonSearch = SamplingSearch(
    scale=2.0, sampler="Halton", scrambled=True
).set_name("Recentering20ScrHaltonSearch", register=True)
Recentering20ScrHammersleySearch = SamplingSearch(
    scale=2.0, sampler="Hammersley", scrambled=True
).set_name("Recentering20ScrHammersleySearch", register=True)
Recentering12ScrHaltonSearch = SamplingSearch(
    scale=1.2, sampler="Halton", scrambled=True
).set_name("Recentering12ScrHaltonSearch", register=True)
Recentering12ScrHammersleySearch = SamplingSearch(
    scale=1.2, sampler="Hammersley", scrambled=True
).set_name("Recentering12ScrHammersleySearch", register=True)
CauchyRecentering12ScrHammersleySearch = SamplingSearch(
    cauchy=True, scale=1.2, sampler="Hammersley", scrambled=True
).set_name("CauchyRecentering12ScrHammersleySearch", register=True)
Recentering7ScrHaltonSearch = SamplingSearch(
    scale=0.7, sampler="Halton", scrambled=True
).set_name("Recentering7ScrHaltonSearch", register=True)
Recentering0ScrHammersleySearch = SamplingSearch(
    scale=0.01, sampler="Hammersley", scrambled=True
).set_name("Recentering0ScrHammersleySearch", register=True)
Recentering0ScrHaltonSearch = SamplingSearch(
    scale=0.01, sampler="Halton", scrambled=True
).set_name("Recentering0ScrHaltonSearch", register=True)
ORecentering1ScrHammersleySearch = SamplingSearch(
    opposition_mode="opposite", scale=0.1, sampler="Hammersley", scrambled=True
).set_name("ORecentering1ScrHammersleySearch", register=True)
ORecentering4ScrHammersleySearch = SamplingSearch(
    opposition_mode="opposite", scale=0.4, sampler="Hammersley", scrambled=True
).set_name("ORecentering4ScrHammersleySearch", register=True)
QORecentering4ScrHammersleySearch = SamplingSearch(
    opposition_mode="quasi", scale=0.4, sampler="Hammersley", scrambled=True
).set_name("QORecentering4ScrHammersleySearch", register=True)
ORecentering1ScrHaltonSearch = SamplingSearch(
    opposition_mode="opposite", scale=0.1, sampler="Halton", scrambled=True
).set_name("ORecentering1ScrHaltonSearch", register=True)
ORecentering4ScrHaltonSearch = SamplingSearch(
    opposition_mode="opposite", scale=0.4, sampler="Halton", scrambled=True
).set_name("ORecentering4ScrHaltonSearch", register=True)
ORecentering7ScrHammersleySearch = SamplingSearch(
    opposition_mode="opposite", scale=0.7, sampler="Hammersley", scrambled=True
).set_name("ORecentering7ScrHammersleySearch", register=True)
QORecentering7ScrHammersleySearch = SamplingSearch(
    opposition_mode="quasi", scale=0.7, sampler="Hammersley", scrambled=True
).set_name("QORecentering7ScrHammersleySearch", register=True)
ORecentering20ScrHaltonSearch = SamplingSearch(
    opposition_mode="opposite", scale=2.0, sampler="Halton", scrambled=True
).set_name("ORecentering20ScrHaltonSearch", register=True)
ORecentering20ScrHammersleySearch = SamplingSearch(
    opposition_mode="opposite", scale=2.0, sampler="Hammersley", scrambled=True
).set_name("ORecentering20ScrHammersleySearch", register=True)
ORecentering12ScrHaltonSearch = SamplingSearch(
    opposition_mode="opposite", scale=1.2, sampler="Halton", scrambled=True
).set_name("ORecentering12ScrHaltonSearch", register=True)
ORecentering12ScrHammersleySearch = SamplingSearch(
    opposition_mode="opposite", scale=1.2, sampler="Hammersley", scrambled=True
).set_name("ORecentering12ScrHammersleySearch", register=True)
ORecentering7ScrHaltonSearch = SamplingSearch(
    opposition_mode="opposite", scale=0.7, sampler="Halton", scrambled=True
).set_name("ORecentering7ScrHaltonSearch", register=True)
ORecentering0ScrHammersleySearch = SamplingSearch(
    opposition_mode="opposite", scale=0.01, sampler="Hammersley", scrambled=True
).set_name("ORecentering0ScrHammersleySearch", register=True)
ORecentering0ScrHaltonSearch = SamplingSearch(
    opposition_mode="opposite", scale=0.01, sampler="Halton", scrambled=True
).set_name("ORecentering0ScrHaltonSearch", register=True)


# Chaining
chainCMASQP = Chaining([CMA, SQP], ["half"]).set_name("chainCMASQP", register=True)
chainCMASQP.no_parallelization = True
chainDEwithR = Chaining([RandomSearch, DE], ["num_workers"]).set_name(
    "chainDEwithR", register=True
)
chainDEwithRsqrt = Chaining([RandomSearch, DE], ["sqrt"]).set_name(
    "chainDEwithRsqrt", register=True
)
chainDEwithRdim = Chaining([RandomSearch, DE], ["dimension"]).set_name(
    "chainDEwithRdim", register=True
)
chainDEwithR30 = Chaining([RandomSearch, DE], [30]).set_name(
    "chainDEwithR30", register=True
)
chainDEwithLHS = Chaining([LHSSearch, DE], ["num_workers"]).set_name(
    "chainDEwithLHS", register=True
)
chainDEwithLHSsqrt = Chaining([LHSSearch, DE], ["sqrt"]).set_name(
    "chainDEwithLHSsqrt", register=True
)
chainDEwithLHSdim = Chaining([LHSSearch, DE], ["dimension"]).set_name(
    "chainDEwithLHSdim", register=True
)
chainDEwithLHS30 = Chaining([LHSSearch, DE], [30]).set_name(
    "chainDEwithLHS30", register=True
)
chainDEwithMetaRecentering = Chaining([MetaRecentering, DE], ["num_workers"]).set_name(
    "chainDEwithMetaRecentering", register=True
)
chainDEwithMetaRecenteringsqrt = Chaining([MetaRecentering, DE], ["sqrt"]).set_name(
    "chainDEwithMetaRecenteringsqrt", register=True
)
chainDEwithMetaRecenteringdim = Chaining([MetaRecentering, DE], ["dimension"]).set_name(
    "chainDEwithMetaRecenteringdim", register=True
)
chainDEwithMetaRecentering30 = Chaining([MetaRecentering, DE], [30]).set_name(
    "chainDEwithMetaRecentering30", register=True
)

chainBOwithR = Chaining([RandomSearch, BO], ["num_workers"]).set_name(
    "chainBOwithR", register=True
)
chainBOwithRsqrt = Chaining([RandomSearch, BO], ["sqrt"]).set_name(
    "chainBOwithRsqrt", register=True
)
chainBOwithRdim = Chaining([RandomSearch, BO], ["dimension"]).set_name(
    "chainBOwithRdim", register=True
)
chainBOwithR30 = Chaining([RandomSearch, BO], [30]).set_name(
    "chainBOwithR30", register=True
)
chainBOwithLHS30 = Chaining([LHSSearch, BO], [30]).set_name(
    "chainBOwithLHS30", register=True
)
chainBOwithLHSsqrt = Chaining([LHSSearch, BO], ["sqrt"]).set_name(
    "chainBOwithLHSsqrt", register=True
)
chainBOwithLHSdim = Chaining([LHSSearch, BO], ["dimension"]).set_name(
    "chainBOwithLHSdim", register=True
)
chainBOwithLHS = Chaining([LHSSearch, BO], ["num_workers"]).set_name(
    "chainBOwithLHS", register=True
)
chainBOwithMetaRecentering30 = Chaining([MetaRecentering, BO], [30]).set_name(
    "chainBOwithMetaRecentering30", register=True
)
chainBOwithMetaRecenteringsqrt = Chaining([MetaRecentering, BO], ["sqrt"]).set_name(
    "chainBOwithMetaRecenteringsqrt", register=True
)
chainBOwithMetaRecenteringdim = Chaining([MetaRecentering, BO], ["dimension"]).set_name(
    "chainBOwithMetaRecenteringdim", register=True
)
chainBOwithMetaRecentering = Chaining([MetaRecentering, BO], ["num_workers"]).set_name(
    "chainBOwithMetaRecentering", register=True
)

chainPSOwithR = Chaining([RandomSearch, PSO], ["num_workers"]).set_name(
    "chainPSOwithR", register=True
)
chainPSOwithRsqrt = Chaining([RandomSearch, PSO], ["sqrt"]).set_name(
    "chainPSOwithRsqrt", register=True
)
chainPSOwithRdim = Chaining([RandomSearch, PSO], ["dimension"]).set_name(
    "chainPSOwithRdim", register=True
)
chainPSOwithR30 = Chaining([RandomSearch, PSO], [30]).set_name(
    "chainPSOwithR30", register=True
)
chainPSOwithLHS30 = Chaining([LHSSearch, PSO], [30]).set_name(
    "chainPSOwithLHS30", register=True
)
chainPSOwithLHSsqrt = Chaining([LHSSearch, PSO], ["sqrt"]).set_name(
    "chainPSOwithLHSsqrt", register=True
)
chainPSOwithLHSdim = Chaining([LHSSearch, PSO], ["dimension"]).set_name(
    "chainPSOwithLHSdim", register=True
)
chainPSOwithLHS = Chaining([LHSSearch, PSO], ["num_workers"]).set_name(
    "chainPSOwithLHS", register=True
)
chainPSOwithMetaRecentering30 = Chaining([MetaRecentering, PSO], [30]).set_name(
    "chainPSOwithMetaRecentering30", register=True
)
chainPSOwithMetaRecenteringsqrt = Chaining([MetaRecentering, PSO], ["sqrt"]).set_name(
    "chainPSOwithMetaRecenteringsqrt", register=True
)
chainPSOwithMetaRecenteringdim = Chaining(
    [MetaRecentering, PSO], ["dimension"]
).set_name("chainPSOwithMetaRecenteringdim", register=True)
chainPSOwithMetaRecentering = Chaining(
    [MetaRecentering, PSO], ["num_workers"]
).set_name("chainPSOwithMetaRecentering", register=True)

chainCMAwithR = Chaining([RandomSearch, CMA], ["num_workers"]).set_name(
    "chainCMAwithR", register=True
)
chainCMAwithRsqrt = Chaining([RandomSearch, CMA], ["sqrt"]).set_name(
    "chainCMAwithRsqrt", register=True
)
chainCMAwithRdim = Chaining([RandomSearch, CMA], ["dimension"]).set_name(
    "chainCMAwithRdim", register=True
)
chainCMAwithR30 = Chaining([RandomSearch, CMA], [30]).set_name(
    "chainCMAwithR30", register=True
)
chainCMAwithLHS30 = Chaining([LHSSearch, CMA], [30]).set_name(
    "chainCMAwithLHS30", register=True
)
chainCMAwithLHSsqrt = Chaining([LHSSearch, CMA], ["sqrt"]).set_name(
    "chainCMAwithLHSsqrt", register=True
)
chainCMAwithLHSdim = Chaining([LHSSearch, CMA], ["dimension"]).set_name(
    "chainCMAwithLHSdim", register=True
)
chainCMAwithLHS = Chaining([LHSSearch, CMA], ["num_workers"]).set_name(
    "chainCMAwithLHS", register=True
)
chainCMAwithMetaRecentering30 = Chaining([MetaRecentering, CMA], [30]).set_name(
    "chainCMAwithMetaRecentering30", register=True
)
chainCMAwithMetaRecenteringsqrt = Chaining([MetaRecentering, CMA], ["sqrt"]).set_name(
    "chainCMAwithMetaRecenteringsqrt", register=True
)
chainCMAwithMetaRecenteringdim = Chaining(
    [MetaRecentering, CMA], ["dimension"]
).set_name("chainCMAwithMetaRecenteringdim", register=True)
chainCMAwithMetaRecentering = Chaining(
    [MetaRecentering, CMA], ["num_workers"]
).set_name("chainCMAwithMetaRecentering", register=True)

# SplitOptimizer
SplitOptimizer3 = ConfSplitOptimizer(num_optims=3).set_name(
    "SplitOptimizer3", register=True
)
SplitOptimizer5 = ConfSplitOptimizer(num_optims=5).set_name(
    "SplitOptimizer5", register=True
)
SplitOptimizer9 = ConfSplitOptimizer(num_optims=9).set_name(
    "SplitOptimizer9", register=True
)
SplitOptimizer13 = ConfSplitOptimizer(num_optims=13).set_name(
    "SplitOptimizer13", register=True
)

# ProgOptimizer
ProgOptimizer3 = ConfSplitOptimizer(num_optims=3, progressive=True).set_name(
    "ProgOptimizer3", register=True
)
ProgOptimizer5 = ConfSplitOptimizer(num_optims=5, progressive=True).set_name(
    "ProgOptimizer5", register=True
)
ProgOptimizer9 = ConfSplitOptimizer(num_optims=9, progressive=True).set_name(
    "ProgOptimizer9", register=True
)
ProgOptimizer13 = ConfSplitOptimizer(num_optims=13, progressive=True).set_name(
    "ProgOptimizer13", register=True
)


# Random search
Zero = RandomSearchMaker(scale=0.0).set_name("Zero", register=True)
LargerScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(
    middle_point=True, scale=500.0
).set_name("LargerScaleRandomSearchPlusMiddlePoint", register=True)
SmallScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(
    middle_point=True, scale=0.01
).set_name("SmallScaleRandomSearchPlusMiddlePoint", register=True)
StupidRandom = RandomSearchMaker(stupid=True).set_name("StupidRandom", register=True)
CauchyRandomSearch = RandomSearchMaker(cauchy=True).set_name(
    "CauchyRandomSearch", register=True
)
RandomScaleRandomSearch = RandomSearchMaker(scale="random", middle_point=True).set_name(
    "RandomScaleRandomSearch", register=True
)
RandomScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(
    scale="random", middle_point=True
).set_name("RandomScaleRandomSearchPlusMiddlePoint", register=True)

# quasi-random
RescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, rescaled=True
).set_name("RescaleScrHammersleySearch", register=True)
LargeHammersleySearch = SamplingSearch(scale=100.0, sampler="Hammersley").set_name(
    "LargeHammersleySearch", register=True
)
LargeScrHammersleySearch = SamplingSearch(
    scale=100.0, sampler="Hammersley", scrambled=True
).set_name("LargeScrHammersleySearch", register=True)
LargeHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=100.0, sampler="Hammersley", middle_point=True
).set_name("LargeHammersleySearchPlusMiddlePoint", register=True)
SmallHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=0.01, sampler="Hammersley", middle_point=True
).set_name("SmallHammersleySearchPlusMiddlePoint", register=True)
LargeScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=100.0, sampler="Hammersley", middle_point=True
).set_name("LargeScrHammersleySearchPlusMiddlePoint", register=True)
SmallScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=0.01, sampler="Hammersley", middle_point=True
).set_name("SmallScrHammersleySearchPlusMiddlePoint", register=True)
LargeScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100.0, middle_point=True, scrambled=True
).set_name("LargeScrHaltonSearchPlusMiddlePoint", register=True)
SmallScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=0.01, middle_point=True, scrambled=True
).set_name("SmallScrHaltonSearchPlusMiddlePoint", register=True)
LargeScrHaltonSearch = SamplingSearch(scale=100.0, scrambled=True).set_name(
    "LargeScrHaltonSearch", register=True
)
LargeHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100.0, middle_point=True
).set_name("LargeHaltonSearchPlusMiddlePoint", register=True)
SmallHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=0.01, middle_point=True
).set_name("SmallHaltonSearchPlusMiddlePoint", register=True)
AvgHaltonSearch = SamplingSearch(recommendation_rule="average_of_best").set_name(
    "AvgHaltonSearch", register=True
)
AvgHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, recommendation_rule="average_of_best"
).set_name("AvgHaltonSearchPlusMiddlePoint", register=True)
AvgLargeHaltonSearch = SamplingSearch(
    scale=100.0, recommendation_rule="average_of_best"
).set_name("AvgLargeHaltonSearch", register=True)
AvgLargeScrHaltonSearch = SamplingSearch(
    scale=100.0, scrambled=True, recommendation_rule="average_of_best"
).set_name("AvgLargeScrHaltonSearch", register=True)
AvgLargeHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100.0, middle_point=True, recommendation_rule="average_of_best"
).set_name("AvgLargeHaltonSearchPlusMiddlePoint", register=True)
AvgSmallHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=0.01, middle_point=True, recommendation_rule="average_of_best"
).set_name("AvgSmallHaltonSearchPlusMiddlePoint", register=True)
AvgScrHaltonSearch = SamplingSearch(
    scrambled=True, recommendation_rule="average_of_best"
).set_name("AvgScrHaltonSearch", register=True)
AvgScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True, recommendation_rule="average_of_best"
).set_name("AvgScrHaltonSearchPlusMiddlePoint", register=True)
AvgLargeScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100.0,
    middle_point=True,
    scrambled=True,
    recommendation_rule="average_of_best",
).set_name("AvgLargeScrHaltonSearchPlusMiddlePoint", register=True)
AvgSmallScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=0.01, middle_point=True, scrambled=True, recommendation_rule="average_of_best"
).set_name("AvgSmallScrHaltonSearchPlusMiddlePoint", register=True)
AvgHammersleySearch = SamplingSearch(
    sampler="Hammersley", recommendation_rule="average_of_best"
).set_name("AvgHammersleySearch", register=True)
AvgHammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best"
).set_name("AvgHammersleySearchPlusMiddlePoint", register=True)
AvgLargeHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=100.0,
    sampler="Hammersley",
    middle_point=True,
    recommendation_rule="average_of_best",
).set_name("AvgLargeHammersleySearchPlusMiddlePoint", register=True)
AvgSmallHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=0.01,
    sampler="Hammersley",
    middle_point=True,
    recommendation_rule="average_of_best",
).set_name("AvgSmallHammersleySearchPlusMiddlePoint", register=True)
AvgLargeScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True,
    scale=100.0,
    sampler="Hammersley",
    middle_point=True,
    recommendation_rule="average_of_best",
).set_name("AvgLargeScrHammersleySearchPlusMiddlePoint", register=True)
AvgSmallScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True,
    scale=0.01,
    sampler="Hammersley",
    middle_point=True,
    recommendation_rule="average_of_best",
).set_name("AvgSmallScrHammersleySearchPlusMiddlePoint", register=True)
AvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True,
    sampler="Hammersley",
    middle_point=True,
    recommendation_rule="average_of_best",
).set_name("AvgScrHammersleySearchPlusMiddlePoint", register=True)
AvgLargeHammersleySearch = SamplingSearch(
    scale=100.0, sampler="Hammersley", recommendation_rule="average_of_best"
).set_name("AvgLargeHammersleySearch", register=True)
AvgLargeScrHammersleySearch = SamplingSearch(
    scale=100.0,
    sampler="Hammersley",
    scrambled=True,
    recommendation_rule="average_of_best",
).set_name("AvgLargeScrHammersleySearch", register=True)
AvgScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, recommendation_rule="average_of_best"
).set_name("AvgScrHammersleySearch", register=True)
AvgRescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley",
    scrambled=True,
    rescaled=True,
    recommendation_rule="average_of_best",
).set_name("AvgRescaleScrHammersleySearch", register=True)
AvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True,
    sampler="Hammersley",
    scrambled=True,
    recommendation_rule="average_of_best",
).set_name("AvgCauchyScrHammersleySearch", register=True)
AvgLHSSearch = SamplingSearch(
    sampler="LHS", recommendation_rule="average_of_best"
).set_name("AvgLHSSearch", register=True)
AvgCauchyLHSSearch = SamplingSearch(
    sampler="LHS", cauchy=True, recommendation_rule="average_of_best"
).set_name("AvgCauchyLHSSearch", register=True)

# Recommendation rule = average of k best, with k = n / 2^d.
EAvgHaltonSearch = SamplingSearch(recommendation_rule="average_of_exp_best").set_name("EAvgHaltonSearch", register=True)
EAvgHaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True, recommendation_rule="average_of_exp_best").set_name(
    "EAvgHaltonSearchPlusMiddlePoint", register=True)
EAvgLargeHaltonSearch = SamplingSearch(scale=100., recommendation_rule="average_of_exp_best").set_name("EAvgLargeHaltonSearch", register=True)
EAvgLargeScrHaltonSearch = SamplingSearch(scale=100., scrambled=True, recommendation_rule="average_of_exp_best").set_name(
    "EAvgLargeScrHaltonSearch", register=True)
EAvgLargeHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgLargeHaltonSearchPlusMiddlePoint", register=True)
EAvgSmallHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgSmallHaltonSearchPlusMiddlePoint", register=True)
EAvgScrHaltonSearch = SamplingSearch(scrambled=True, recommendation_rule="average_of_exp_best").set_name("EAvgScrHaltonSearch", register=True)
EAvgScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True, recommendation_rule="average_of_exp_best").set_name("EAvgScrHaltonSearchPlusMiddlePoint", register=True)
EAvgLargeScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, scrambled=True, recommendation_rule="average_of_exp_best").set_name("EAvgLargeScrHaltonSearchPlusMiddlePoint", register=True)
EAvgSmallScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, scrambled=True, recommendation_rule="average_of_exp_best").set_name("EAvgSmallScrHaltonSearchPlusMiddlePoint", register=True)
EAvgHammersleySearch = SamplingSearch(sampler="Hammersley", recommendation_rule="average_of_exp_best").set_name(
    "EAvgHammersleySearch", register=True)
EAvgHammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgHammersleySearchPlusMiddlePoint", register=True)
EAvgLargeHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=100., sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgLargeHammersleySearchPlusMiddlePoint", register=True)
EAvgSmallHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=.01, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgSmallHammersleySearchPlusMiddlePoint", register=True)
EAvgLargeScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=100., sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgLargeScrHammersleySearchPlusMiddlePoint", register=True)
EAvgSmallScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=.01, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgSmallScrHammersleySearchPlusMiddlePoint", register=True)
EAvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("EAvgScrHammersleySearchPlusMiddlePoint", register=True)
EAvgLargeHammersleySearch = SamplingSearch(scale=100., sampler="Hammersley",
                                          recommendation_rule="average_of_exp_best").set_name("EAvgLargeHammersleySearch", register=True)
EAvgLargeScrHammersleySearch = SamplingSearch(
    scale=100., sampler="Hammersley", scrambled=True, recommendation_rule="average_of_exp_best").set_name("EAvgLargeScrHammersleySearch", register=True)
EAvgScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                        recommendation_rule="average_of_exp_best").set_name("EAvgScrHammersleySearch", register=True)
EAvgRescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, rescaled=True, recommendation_rule="average_of_exp_best").set_name("EAvgRescaleScrHammersleySearch", register=True)
EAvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_exp_best").set_name("EAvgCauchyScrHammersleySearch", register=True)
EAvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_exp_best").set_name("EAvgLHSSearch", register=True)
EAvgCauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True, recommendation_rule="average_of_exp_best").set_name(
    "EAvgCauchyLHSSearch", register=True)

# Recommendation rule = by convex hull.
CHAvgHaltonSearch = SamplingSearch(recommendation_rule="average_of_hull_best").set_name("CHAvgHaltonSearch", register=True)
CHAvgHaltonSearchPlusMiddlePoint = SamplingSearch(middle_point=True, recommendation_rule="average_of_hull_best").set_name(
    "CHAvgHaltonSearchPlusMiddlePoint", register=True)
CHAvgLargeHaltonSearch = SamplingSearch(scale=100., recommendation_rule="average_of_hull_best").set_name("CHAvgLargeHaltonSearch", register=True)
CHAvgLargeScrHaltonSearch = SamplingSearch(scale=100., scrambled=True, recommendation_rule="average_of_hull_best").set_name(
    "CHAvgLargeScrHaltonSearch", register=True)
CHAvgLargeHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgLargeHaltonSearchPlusMiddlePoint", register=True)
CHAvgSmallHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgSmallHaltonSearchPlusMiddlePoint", register=True)
CHAvgScrHaltonSearch = SamplingSearch(scrambled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgScrHaltonSearch", register=True)
CHAvgScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgScrHaltonSearchPlusMiddlePoint", register=True)
CHAvgLargeScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=100., middle_point=True, scrambled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgLargeScrHaltonSearchPlusMiddlePoint", register=True)
CHAvgSmallScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    scale=.01, middle_point=True, scrambled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgSmallScrHaltonSearchPlusMiddlePoint", register=True)
CHAvgHammersleySearch = SamplingSearch(sampler="Hammersley", recommendation_rule="average_of_hull_best").set_name(
    "CHAvgHammersleySearch", register=True)
CHAvgHammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgHammersleySearchPlusMiddlePoint", register=True)
CHAvgLargeHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=100., sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgLargeHammersleySearchPlusMiddlePoint", register=True)
CHAvgSmallHammersleySearchPlusMiddlePoint = SamplingSearch(
    scale=.01, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgSmallHammersleySearchPlusMiddlePoint", register=True)
CHAvgLargeScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=100., sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgLargeScrHammersleySearchPlusMiddlePoint", register=True)
CHAvgSmallScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, scale=.01, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgSmallScrHammersleySearchPlusMiddlePoint", register=True)
CHAvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("CHAvgScrHammersleySearchPlusMiddlePoint", register=True)
CHAvgLargeHammersleySearch = SamplingSearch(scale=100., sampler="Hammersley",
                                          recommendation_rule="average_of_hull_best").set_name("CHAvgLargeHammersleySearch", register=True)
CHAvgLargeScrHammersleySearch = SamplingSearch(
    scale=100., sampler="Hammersley", scrambled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgLargeScrHammersleySearch", register=True)
CHAvgScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                        recommendation_rule="average_of_hull_best").set_name("CHAvgScrHammersleySearch", register=True)
CHAvgRescaleScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, rescaled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgRescaleScrHammersleySearch", register=True)
CHAvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_hull_best").set_name("CHAvgCauchyScrHammersleySearch", register=True)
CHAvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_hull_best").set_name("CHAvgLHSSearch", register=True)
CHAvgCauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True, recommendation_rule="average_of_hull_best").set_name(
    "CHAvgCauchyLHSSearch", register=True)
