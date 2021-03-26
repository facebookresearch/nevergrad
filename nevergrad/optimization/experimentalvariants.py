# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .oneshot import SamplingSearch
from .differentialevolution import DifferentialEvolution
from .optimizerlib import RandomSearchMaker, SQP, LHSSearch, DE, RandomSearch, MetaRecentering, MetaTuneRecentering  # type: ignore
from .optimizerlib import (
    MultipleSingleRuns,
    ParametrizedOnePlusOne,
    ParametrizedCMA,
    ConfSplitOptimizer,
    ParametrizedBO,
    EMNA,
    NGOpt10,
)
from .optimizerlib import CMA, Chaining, PSO, BO

# DE
OnePointDE = DifferentialEvolution(crossover="onepoint").set_name("OnePointDE", register=True)
ParametrizationDE = DifferentialEvolution(crossover="parametrization").set_name(
    "ParametrizationDE", register=True
)
MiniDE = DifferentialEvolution(initialization="gaussian", scale="mini").set_name("MiniDE", register=True)
MiniLhsDE = DifferentialEvolution(initialization="LHS", scale="mini").set_name("MiniLhsDE", register=True)
MiniQrDE = DifferentialEvolution(initialization="QR", scale="mini").set_name("MiniQrDE", register=True)
AlmostRotationInvariantDEAndBigPop = DifferentialEvolution(crossover=0.9, popsize="dimension").set_name(
    "AlmostRotationInvariantDEAndBigPop", register=True
)
BPRotationInvariantDE = DifferentialEvolution(crossover=1.0, popsize="large").set_name(
    "BPRotationInvariantDE", register=True
)

# CMA
MilliCMA = ParametrizedCMA(scale=1e-3).set_name("MilliCMA", register=True)
MicroCMA = ParametrizedCMA(scale=1e-6).set_name("MicroCMA", register=True)
FCMAs03 = ParametrizedCMA(fcmaes=True, scale=0.3).set_name("FCMAs03", register=True)
FCMAp13 = ParametrizedCMA(fcmaes=True, scale=0.1, popsize=13).set_name("FCMAp13", register=True)
ECMA = ParametrizedCMA(elitist=True).set_name("ECMA", register=True)

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
FastGANoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="random", mutation="fastga").set_name(
    "FastGANoisyDiscreteOnePlusOne", register=True
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
MidQRBO = ParametrizedBO(initialization="Hammersley", middle_point=True).set_name("MidQRBO", register=True)
LBO = ParametrizedBO(initialization="LHS").set_name("LBO", register=True)

# EMNA
IsoEMNA = EMNA(naive=False).set_name("IsoEMNA", register=True)
NaiveAnisoEMNA = EMNA(isotropic=False).set_name("NaiveAnisoEMNA", register=True)
AnisoEMNA = EMNA(naive=False, isotropic=False).set_name("AnisoEMNA", register=True)
IsoEMNATBPSA = EMNA(naive=False, population_size_adaptation=True).set_name("IsoEMNATBPSA", register=True)
NaiveIsoEMNATBPSA = EMNA(population_size_adaptation=True).set_name("NaiveIsoEMNATBPSA", register=True)
AnisoEMNATBPSA = EMNA(naive=False, isotropic=False, population_size_adaptation=True).set_name(
    "AnisoEMNATBPSA", register=True
)
NaiveAnisoEMNATBPSA = EMNA(isotropic=False, population_size_adaptation=True).set_name(
    "NaiveAnisoEMNATBPSA", register=True
)

# Recentering
MetaCauchyRecentering = SamplingSearch(
    cauchy=True, autorescale=True, sampler="Hammersley", scrambled=True
).set_name("MetaCauchyRecentering", register=True)
# Chaining
ChainCMASQP = Chaining([CMA, SQP], ["half"]).set_name("ChainCMASQP", register=True)
ChainCMASQP.no_parallelization = True
ChainDEwithR = Chaining([RandomSearch, DE], ["num_workers"]).set_name("ChainDEwithR", register=True)
ChainDEwithRsqrt = Chaining([RandomSearch, DE], ["sqrt"]).set_name("ChainDEwithRsqrt", register=True)
ChainDEwithRdim = Chaining([RandomSearch, DE], ["dimension"]).set_name("ChainDEwithRdim", register=True)
ChainDEwithR30 = Chaining([RandomSearch, DE], [30]).set_name("ChainDEwithR30", register=True)
ChainDEwithLHS = Chaining([LHSSearch, DE], ["num_workers"]).set_name("ChainDEwithLHS", register=True)
ChainDEwithLHSsqrt = Chaining([LHSSearch, DE], ["sqrt"]).set_name("ChainDEwithLHSsqrt", register=True)
ChainDEwithLHSdim = Chaining([LHSSearch, DE], ["dimension"]).set_name("ChainDEwithLHSdim", register=True)
ChainDEwithLHS30 = Chaining([LHSSearch, DE], [30]).set_name("ChainDEwithLHS30", register=True)
ChainDEwithMetaRecentering = Chaining([MetaRecentering, DE], ["num_workers"]).set_name(
    "ChainDEwithMetaRecentering", register=True
)
ChainDEwithMetaRecenteringsqrt = Chaining([MetaRecentering, DE], ["sqrt"]).set_name(
    "ChainDEwithMetaRecenteringsqrt", register=True
)
ChainDEwithMetaRecenteringdim = Chaining([MetaRecentering, DE], ["dimension"]).set_name(
    "ChainDEwithMetaRecenteringdim", register=True
)
ChainDEwithMetaRecentering30 = Chaining([MetaRecentering, DE], [30]).set_name(
    "ChainDEwithMetaRecentering30", register=True
)
ChainBOwithMetaTuneRecentering = Chaining([MetaTuneRecentering, BO], ["num_workers"]).set_name(
    "ChainBOwithMetaTuneRecentering", register=True
)
ChainBOwithMetaTuneRecenteringsqrt = Chaining([MetaTuneRecentering, BO], ["sqrt"]).set_name(
    "ChainBOwithMetaTuneRecenteringsqrt", register=True
)
ChainBOwithMetaTuneRecenteringdim = Chaining([MetaTuneRecentering, BO], ["dimension"]).set_name(
    "ChainBOwithMetaTuneRecenteringdim", register=True
)
ChainBOwithMetaTuneRecentering30 = Chaining([MetaTuneRecentering, BO], [30]).set_name(
    "ChainBOwithMetaTuneRecentering30", register=True
)

ChainDEwithMetaTuneRecentering = Chaining([MetaTuneRecentering, DE], ["num_workers"]).set_name(
    "ChainDEwithMetaTuneRecentering", register=True
)
ChainDEwithMetaTuneRecenteringsqrt = Chaining([MetaTuneRecentering, DE], ["sqrt"]).set_name(
    "ChainDEwithMetaTuneRecenteringsqrt", register=True
)
ChainDEwithMetaTuneRecenteringdim = Chaining([MetaTuneRecentering, DE], ["dimension"]).set_name(
    "ChainDEwithMetaTuneRecenteringdim", register=True
)
ChainDEwithMetaTuneRecentering30 = Chaining([MetaTuneRecentering, DE], [30]).set_name(
    "ChainDEwithMetaTuneRecentering30", register=True
)


ChainBOwithR = Chaining([RandomSearch, BO], ["num_workers"]).set_name("ChainBOwithR", register=True)
ChainBOwithRsqrt = Chaining([RandomSearch, BO], ["sqrt"]).set_name("ChainBOwithRsqrt", register=True)
ChainBOwithRdim = Chaining([RandomSearch, BO], ["dimension"]).set_name("ChainBOwithRdim", register=True)
ChainBOwithR30 = Chaining([RandomSearch, BO], [30]).set_name("ChainBOwithR30", register=True)
ChainBOwithLHS30 = Chaining([LHSSearch, BO], [30]).set_name("ChainBOwithLHS30", register=True)
ChainBOwithLHSsqrt = Chaining([LHSSearch, BO], ["sqrt"]).set_name("ChainBOwithLHSsqrt", register=True)
ChainBOwithLHSdim = Chaining([LHSSearch, BO], ["dimension"]).set_name("ChainBOwithLHSdim", register=True)
ChainBOwithLHS = Chaining([LHSSearch, BO], ["num_workers"]).set_name("ChainBOwithLHS", register=True)
ChainBOwithMetaRecentering30 = Chaining([MetaRecentering, BO], [30]).set_name(
    "ChainBOwithMetaRecentering30", register=True
)
ChainBOwithMetaRecenteringsqrt = Chaining([MetaRecentering, BO], ["sqrt"]).set_name(
    "ChainBOwithMetaRecenteringsqrt", register=True
)
ChainBOwithMetaRecenteringdim = Chaining([MetaRecentering, BO], ["dimension"]).set_name(
    "ChainBOwithMetaRecenteringdim", register=True
)
ChainBOwithMetaRecentering = Chaining([MetaRecentering, BO], ["num_workers"]).set_name(
    "ChainBOwithMetaRecentering", register=True
)

ChainPSOwithR = Chaining([RandomSearch, PSO], ["num_workers"]).set_name("ChainPSOwithR", register=True)
ChainPSOwithRsqrt = Chaining([RandomSearch, PSO], ["sqrt"]).set_name("ChainPSOwithRsqrt", register=True)
ChainPSOwithRdim = Chaining([RandomSearch, PSO], ["dimension"]).set_name("ChainPSOwithRdim", register=True)
ChainPSOwithR30 = Chaining([RandomSearch, PSO], [30]).set_name("ChainPSOwithR30", register=True)
ChainPSOwithLHS30 = Chaining([LHSSearch, PSO], [30]).set_name("ChainPSOwithLHS30", register=True)
ChainPSOwithLHSsqrt = Chaining([LHSSearch, PSO], ["sqrt"]).set_name("ChainPSOwithLHSsqrt", register=True)
ChainPSOwithLHSdim = Chaining([LHSSearch, PSO], ["dimension"]).set_name("ChainPSOwithLHSdim", register=True)
ChainPSOwithLHS = Chaining([LHSSearch, PSO], ["num_workers"]).set_name("ChainPSOwithLHS", register=True)
ChainPSOwithMetaRecentering30 = Chaining([MetaRecentering, PSO], [30]).set_name(
    "ChainPSOwithMetaRecentering30", register=True
)
ChainPSOwithMetaRecenteringsqrt = Chaining([MetaRecentering, PSO], ["sqrt"]).set_name(
    "ChainPSOwithMetaRecenteringsqrt", register=True
)
ChainPSOwithMetaRecenteringdim = Chaining([MetaRecentering, PSO], ["dimension"]).set_name(
    "ChainPSOwithMetaRecenteringdim", register=True
)
ChainPSOwithMetaRecentering = Chaining([MetaRecentering, PSO], ["num_workers"]).set_name(
    "ChainPSOwithMetaRecentering", register=True
)

ChainCMAwithR = Chaining([RandomSearch, CMA], ["num_workers"]).set_name("ChainCMAwithR", register=True)
ChainCMAwithRsqrt = Chaining([RandomSearch, CMA], ["sqrt"]).set_name("ChainCMAwithRsqrt", register=True)
ChainCMAwithRdim = Chaining([RandomSearch, CMA], ["dimension"]).set_name("ChainCMAwithRdim", register=True)
ChainCMAwithR30 = Chaining([RandomSearch, CMA], [30]).set_name("ChainCMAwithR30", register=True)
ChainCMAwithLHS30 = Chaining([LHSSearch, CMA], [30]).set_name("ChainCMAwithLHS30", register=True)
ChainCMAwithLHSsqrt = Chaining([LHSSearch, CMA], ["sqrt"]).set_name("ChainCMAwithLHSsqrt", register=True)
ChainCMAwithLHSdim = Chaining([LHSSearch, CMA], ["dimension"]).set_name("ChainCMAwithLHSdim", register=True)
ChainCMAwithLHS = Chaining([LHSSearch, CMA], ["num_workers"]).set_name("ChainCMAwithLHS", register=True)
ChainCMAwithMetaRecentering30 = Chaining([MetaRecentering, CMA], [30]).set_name(
    "ChainCMAwithMetaRecentering30", register=True
)
ChainCMAwithMetaRecenteringsqrt = Chaining([MetaRecentering, CMA], ["sqrt"]).set_name(
    "ChainCMAwithMetaRecenteringsqrt", register=True
)
ChainCMAwithMetaRecenteringdim = Chaining([MetaRecentering, CMA], ["dimension"]).set_name(
    "ChainCMAwithMetaRecenteringdim", register=True
)
ChainCMAwithMetaRecentering = Chaining([MetaRecentering, CMA], ["num_workers"]).set_name(
    "ChainCMAwithMetaRecentering", register=True
)

# Random search
Zero = RandomSearchMaker(scale=0.0).set_name("Zero", register=True)
StupidRandom = RandomSearchMaker(stupid=True).set_name("StupidRandom", register=True)
CauchyRandomSearch = RandomSearchMaker(sampler="cauchy").set_name("CauchyRandomSearch", register=True)
RandomScaleRandomSearch = RandomSearchMaker(scale="random", middle_point=True).set_name(
    "RandomScaleRandomSearch", register=True
)
RandomScaleRandomSearchPlusMiddlePoint = RandomSearchMaker(scale="random", middle_point=True).set_name(
    "RandomScaleRandomSearchPlusMiddlePoint", register=True
)

# quasi-random
RescaleScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True, rescaled=True).set_name(
    "RescaleScrHammersleySearch", register=True
)
AvgHammersleySearch = SamplingSearch(sampler="Hammersley", recommendation_rule="average_of_best").set_name(
    "AvgHammersleySearch", register=True
)
AvgHammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best"
).set_name("AvgHammersleySearchPlusMiddlePoint", register=True)
HCHAvgRandomSearch = SamplingSearch(sampler="Random", recommendation_rule="average_of_hull_best").set_name(
    "HCHAvgRandomSearch", register=True
)
AvgRandomSearch = SamplingSearch(sampler="Random", recommendation_rule="average_of_best").set_name(
    "AvgRandomSearch", register=True
)

# Recommendation rule = average of k best, with k = n / 2^d.
TEAvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best"
).set_name("TEAvgScrHammersleySearchPlusMiddlePoint", register=True)
TEAvgScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, recommendation_rule="average_of_exp_best"
).set_name("TEAvgScrHammersleySearch", register=True)
TEAvgRandomSearch = SamplingSearch(sampler="Random", recommendation_rule="average_of_exp_best").set_name(
    "TEAvgRandomSearch", register=True
)
TEAvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_exp_best"
).set_name("TEAvgCauchyScrHammersleySearch", register=True)
TEAvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_exp_best").set_name(
    "TEAvgLHSSearch", register=True
)
TEAvgCauchyLHSSearch = SamplingSearch(
    sampler="LHS", cauchy=True, recommendation_rule="average_of_exp_best"
).set_name("TEAvgCauchyLHSSearch", register=True)

# Recommendation rule = by convex hull.
HCHAvgScrHaltonSearch = SamplingSearch(scrambled=True, recommendation_rule="average_of_hull_best").set_name(
    "HCHAvgScrHaltonSearch", register=True
)
HCHAvgScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True, recommendation_rule="average_of_hull_best"
).set_name("HCHAvgScrHaltonSearchPlusMiddlePoint", register=True)
HCHAvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best"
).set_name("HCHAvgScrHammersleySearchPlusMiddlePoint", register=True)
HCHAvgLargeHammersleySearch = SamplingSearch(
    scale=100.0, sampler="Hammersley", recommendation_rule="average_of_hull_best"
).set_name("HCHAvgLargeHammersleySearch", register=True)
HCHAvgScrHammersleySearch = SamplingSearch(
    sampler="Hammersley", scrambled=True, recommendation_rule="average_of_hull_best"
).set_name("HCHAvgScrHammersleySearch", register=True)
HCHAvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_hull_best"
).set_name("HCHAvgCauchyScrHammersleySearch", register=True)
HCHAvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_hull_best").set_name(
    "HCHAvgLHSSearch", register=True
)
HCHAvgCauchyLHSSearch = SamplingSearch(
    sampler="LHS", cauchy=True, recommendation_rule="average_of_hull_best"
).set_name("HCHAvgCauchyLHSSearch", register=True)

# Split on top of competence map.
MetaNGOpt10 = ConfSplitOptimizer(
    multivariate_optimizer=NGOpt10, monovariate_optimizer=NGOpt10, non_deterministic_descriptor=False
).set_name("MetaNGOpt10", register=True)

# Multiple single runs for multi-objective optimization.
NGOptSingle9 = MultipleSingleRuns(num_single_runs=9).set_name("NGOptSingle9", register=True)
NGOptSingle16 = MultipleSingleRuns(num_single_runs=16).set_name("NGOptSingle16", register=True)
NGOptSingle25 = MultipleSingleRuns(num_single_runs=25).set_name("NGOptSingle25", register=True)
