# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .oneshot import SamplingSearch
from .differentialevolution import DifferentialEvolution
from .optimizerlib import RandomSearchMaker, SQP, LHSSearch, DE, RandomSearch, MetaRecentering, MetaTuneRecentering  # type: ignore
from .optimizerlib import (
    OptimisticNoisyOnePlusOne,
    OptimisticDiscreteOnePlusOne,
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
IsoEMNATBPSA = EMNA(naive=False, population_size_adaptation=True).set_name("IsoEMNATBPSA", register=True)
NaiveIsoEMNATBPSA = EMNA(population_size_adaptation=True).set_name("NaiveIsoEMNATBPSA", register=True)
AnisoEMNATBPSA = EMNA(naive=False, isotropic=False, population_size_adaptation=True).set_name("AnisoEMNATBPSA", register=True)
NaiveAnisoEMNATBPSA = EMNA(isotropic=False, population_size_adaptation=True).set_name("NaiveAnisoEMNATBPSA", register=True)

# Recentering
MetaCauchyRecentering = SamplingSearch(
    cauchy=True, autorescale=True, sampler="Hammersley", scrambled=True
).set_name("MetaCauchyRecentering", register=True)
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
chainBOwithMetaTuneRecentering = Chaining([MetaTuneRecentering, BO], ["num_workers"]).set_name(
    "chainBOwithMetaTuneRecentering", register=True
)
chainBOwithMetaTuneRecenteringsqrt = Chaining([MetaTuneRecentering, BO], ["sqrt"]).set_name(
    "chainBOwithMetaTuneRecenteringsqrt", register=True
)
chainBOwithMetaTuneRecenteringdim = Chaining([MetaTuneRecentering, BO], ["dimension"]).set_name(
    "chainBOwithMetaTuneRecenteringdim", register=True
)
chainBOwithMetaTuneRecentering30 = Chaining([MetaTuneRecentering, BO], [30]).set_name(
    "chainBOwithMetaTuneRecentering30", register=True
)

chainDEwithMetaTuneRecentering = Chaining([MetaTuneRecentering, DE], ["num_workers"]).set_name(
    "chainDEwithMetaTuneRecentering", register=True
)
chainDEwithMetaTuneRecenteringsqrt = Chaining([MetaTuneRecentering, DE], ["sqrt"]).set_name(
    "chainDEwithMetaTuneRecenteringsqrt", register=True
)
chainDEwithMetaTuneRecenteringdim = Chaining([MetaTuneRecentering, DE], ["dimension"]).set_name(
    "chainDEwithMetaTuneRecenteringdim", register=True
)
chainDEwithMetaTuneRecentering30 = Chaining([MetaTuneRecentering, DE], [30]).set_name(
    "chainDEwithMetaTuneRecentering30", register=True
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
SplitCMA3 = ConfSplitOptimizer(num_optims=3).set_name(
    "SplitCMA3", register=True
)
SplitCMA5 = ConfSplitOptimizer(num_optims=5).set_name(
    "SplitCMA5", register=True
)
SplitCMA9 = ConfSplitOptimizer(num_optims=9).set_name(
    "SplitCMA9", register=True
)
SplitCMA13 = ConfSplitOptimizer(num_optims=13).set_name(
    "SplitCMA13", register=True
)
SplitCMAAuto = ConfSplitOptimizer().set_name("SplitCMAAuto", register=True)

# ProgOptimizer
ProgONOPO3 = ConfSplitOptimizer(num_optims=3, progressive=True, multivariate_optimizer=OptimisticNoisyOnePlusOne).set_name(
    "ProgONOPO3", register=True
)
ProgONOPO5 = ConfSplitOptimizer(num_optims=5, progressive=True, multivariate_optimizer=OptimisticNoisyOnePlusOne).set_name(
    "ProgONOPO5", register=True
)
ProgONOPO9 = ConfSplitOptimizer(num_optims=9, progressive=True, multivariate_optimizer=OptimisticNoisyOnePlusOne).set_name(
    "ProgONOPO9", register=True
)
ProgONOPO13 = ConfSplitOptimizer(num_optims=13, progressive=True, multivariate_optimizer=OptimisticNoisyOnePlusOne).set_name(
    "ProgONOPO13", register=True
)
ProgONOPOAuto = ConfSplitOptimizer(progressive=True, multivariate_optimizer=OptimisticNoisyOnePlusOne).set_name(
    "ProgONOPOAuto", register=True
)
# ProgOptimizer
ProgODOPO3 = ConfSplitOptimizer(num_optims=3, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne).set_name(
    "ProgODOPO3", register=True
)
ProgODOPO5 = ConfSplitOptimizer(num_optims=5, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne).set_name(
    "ProgODOPO5", register=True
)
ProgODOPO9 = ConfSplitOptimizer(num_optims=9, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne).set_name(
    "ProgODOPO9", register=True
)
ProgODOPO13 = ConfSplitOptimizer(num_optims=13, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne).set_name(
    "ProgODOPO13", register=True
)
ProgODOPOAuto = ConfSplitOptimizer(progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne).set_name(
    "ProgODOPOAuto", register=True
)


# Random search
Zero = RandomSearchMaker(scale=0.0).set_name("Zero", register=True)
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
AvgHammersleySearch = SamplingSearch(
    sampler="Hammersley", recommendation_rule="average_of_best"
).set_name("AvgHammersleySearch", register=True)
AvgHammersleySearchPlusMiddlePoint = SamplingSearch(
    sampler="Hammersley", middle_point=True, recommendation_rule="average_of_best"
).set_name("AvgHammersleySearchPlusMiddlePoint", register=True)
HCHAvgRandomSearch = SamplingSearch(
    sampler="Random", recommendation_rule="average_of_hull_best"
).set_name("HCHAvgRandomSearch", register=True)
AvgRandomSearch = SamplingSearch(
    sampler="Random", recommendation_rule="average_of_best"
).set_name("AvgRandomSearch", register=True)

# Recommendation rule = average of k best, with k = n / 2^d.
TEAvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_exp_best").set_name("TEAvgScrHammersleySearchPlusMiddlePoint", register=True)
TEAvgScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                        recommendation_rule="average_of_exp_best").set_name("TEAvgScrHammersleySearch", register=True)
TEAvgRandomSearch = SamplingSearch(sampler="Random",
                                        recommendation_rule="average_of_exp_best").set_name("TEAvgRandomSearch", register=True)
TEAvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_exp_best").set_name("TEAvgCauchyScrHammersleySearch", register=True)
TEAvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_exp_best").set_name("TEAvgLHSSearch", register=True)
TEAvgCauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True, recommendation_rule="average_of_exp_best").set_name(
    "TEAvgCauchyLHSSearch", register=True)

# Recommendation rule = by convex hull.
HCHAvgScrHaltonSearch = SamplingSearch(scrambled=True, recommendation_rule="average_of_hull_best").set_name("HCHAvgScrHaltonSearch", register=True)
HCHAvgScrHaltonSearchPlusMiddlePoint = SamplingSearch(
    middle_point=True, scrambled=True, recommendation_rule="average_of_hull_best").set_name("HCHAvgScrHaltonSearchPlusMiddlePoint", register=True)
HCHAvgScrHammersleySearchPlusMiddlePoint = SamplingSearch(
    scrambled=True, sampler="Hammersley", middle_point=True, recommendation_rule="average_of_hull_best").set_name("HCHAvgScrHammersleySearchPlusMiddlePoint", register=True)
HCHAvgLargeHammersleySearch = SamplingSearch(scale=100., sampler="Hammersley",
                                          recommendation_rule="average_of_hull_best").set_name("HCHAvgLargeHammersleySearch", register=True)
HCHAvgScrHammersleySearch = SamplingSearch(sampler="Hammersley", scrambled=True,
                                        recommendation_rule="average_of_hull_best").set_name("HCHAvgScrHammersleySearch", register=True)
HCHAvgCauchyScrHammersleySearch = SamplingSearch(
    cauchy=True, sampler="Hammersley", scrambled=True, recommendation_rule="average_of_hull_best").set_name("HCHAvgCauchyScrHammersleySearch", register=True)
HCHAvgLHSSearch = SamplingSearch(sampler="LHS", recommendation_rule="average_of_hull_best").set_name("HCHAvgLHSSearch", register=True)
HCHAvgCauchyLHSSearch = SamplingSearch(sampler="LHS", cauchy=True, recommendation_rule="average_of_hull_best").set_name(
    "HCHAvgCauchyLHSSearch", register=True)


# Old variants of NGOpt (to be deleted at some point ?)
@registry.register
class NGOpt4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing youe own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        # override because it is different from NGOptBase (bug?)
        self.has_discrete_not_softmax = self._has_discrete
        self.fully_continuous = self.fully_continuous and not self.has_discrete_not_softmax and self._arity < 0
        budget, num_workers = self.budget, self.num_workers
        assert budget is not None
        optimClass: base.OptCls
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.descriptors.metrizable):
            mutation = "portfolio" if budget > 1000 else "discrete"
            optimClass = ParametrizedOnePlusOne(crossover=True, mutation=mutation, noise_handling="optimistic")
        elif self._arity > 0:
            if self._arity == 2:
                optimClass = DiscreteOnePlusOne
            else:
                optimClass = AdaptiveDiscreteOnePlusOne if self._arity < 5 else CMandAS2
        else:
            # pylint: disable=too-many-nested-blocks
            if self.has_noise and self.fully_continuous and self.dimension > 100:
                # Waow, this is actually a discrete algorithm.
                optimClass = ConfSplitOptimizer(num_optims=13, progressive=True,
                                                multivariate_optimizer=OptimisticDiscreteOnePlusOne)
            else:
                if self.has_noise and self.fully_continuous:
                    if budget > 100:
                        optimClass = SQP
                    else:
                        # This is the realm of population control. FIXME: should we pair with a bandit ?
                        optimClass = TBPSA
                else:
                    if self.has_discrete_not_softmax or not self.parametrization.descriptors.metrizable or not self.fully_continuous:
                        optimClass = DoubleFastGADiscreteOnePlusOne
                    else:
                        if num_workers > budget / 5:
                            if num_workers > budget / 2. or budget < self.dimension:
                                optimClass = MetaTuneRecentering
                            elif self.dimension < 5 and budget < 100:
                                optimClass = DiagonalCMA
                            elif self.dimension < 5 and budget < 500:
                                optimClass = Chaining([DiagonalCMA, MetaModel], [100])
                            else:
                                optimClass = NaiveTBPSA
                        else:
                            # Possibly a good idea to go memetic for large budget, but something goes wrong for the moment.
                            if num_workers == 1 and budget > 6000 and self.dimension > 7:  # Let us go memetic.
                                optimClass = chainNaiveTBPSACMAPowell
                            else:
                                if num_workers == 1 and budget < self.dimension * 30:
                                    if self.dimension > 30:  # One plus one so good in large ratio "dimension / budget".
                                        optimClass = OnePlusOne
                                    elif self.dimension < 5:
                                        optimClass = MetaModel
                                    else:
                                        optimClass = Cobyla
                                else:
                                    if self.dimension > 2000:  # DE is great in such a case (?).
                                        optimClass = DE
                                    else:
                                        if self.dimension < 10 and budget < 500:
                                            optimClass = MetaModel
                                        else:
                                            if self.dimension > 40 and num_workers > self.dimension and budget < 7 * self.dimension ** 2:
                                                optimClass = DiagonalCMA
                                            elif 3 * num_workers > self.dimension ** 2 and budget > self.dimension ** 2:
                                                optimClass = MetaModel
                                            else:
                                                optimClass = CMA
        return optimClass
