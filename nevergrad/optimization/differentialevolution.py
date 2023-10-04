# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import metamodel
from . import base
from . import oneshot


class Crossover:
    def __init__(
        self,
        random_state: np.random.RandomState,
        crossover: tp.Union[str, float],
        parameter: tp.Optional[p.Parameter] = None,
    ):
        self.CR = 0.5
        self.crossover = crossover
        self.random_state = random_state
        if isinstance(crossover, float):
            self.CR = crossover
        elif crossover == "random":
            self.CR = self.random_state.uniform(0.0, 1.0)
        elif crossover not in ["twopoints", "onepoint", "rotated_twopoints", "voronoi"]:
            raise ValueError(f'Unknown crossover "{crossover}"')
        self.shape = (
            np.array(parameter.value).shape if (crossover == "voronoi" and parameter is not None) else None
        )

    def apply(self, donor: np.ndarray, individual: np.ndarray) -> None:
        dim = donor.size
        if self.crossover == "twopoints" and dim >= 4:
            return self.twopoints(donor, individual)
        elif self.crossover == "rotated_twopoints" and dim >= 4:
            return self.rotated_twopoints(donor, individual)
        elif self.crossover == "onepoint" and dim >= 3:
            return self.onepoint(donor, individual)
        elif self.crossover == "voronoi":
            return self.voronoi(donor, individual)
        else:
            return self.variablewise(donor, individual)

    def variablewise(self, donor: np.ndarray, individual: np.ndarray) -> None:
        R = self.random_state.randint(donor.size)
        # the following could be updated to vectorial uniform sampling (changes recomms)
        transfer = np.array(
            [idx != R and self.random_state.uniform(0, 1) > self.CR for idx in range(donor.size)]
        )
        donor[transfer] = individual[transfer]

    def onepoint(self, donor: np.ndarray, individual: np.ndarray) -> None:
        R = self.random_state.randint(1, donor.size)
        if self.random_state.choice([True, False]):
            donor[R:] = individual[R:]
        else:
            donor[:R] = individual[:R]

    def twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None:
        bounds = sorted(self.random_state.choice(donor.size + 1, size=2, replace=False).tolist())
        if bounds[1] == donor.size and not bounds[0]:  # make sure there is at least one point crossover
            bounds[self.random_state.randint(2)] = self.random_state.randint(1, donor.size)
        if self.random_state.choice([True, False]):
            donor[bounds[0] : bounds[1]] = individual[bounds[0] : bounds[1]]
        else:
            donor[: bounds[0]] = individual[: bounds[0]]
            donor[bounds[1] :] = individual[bounds[1] :]

    def rotated_twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None:
        bounds = sorted(self.random_state.choice(donor.size + 1, size=2, replace=False).tolist())
        if bounds[1] == donor.size and not bounds[0]:  # make sure there is at least one point crossover
            bounds[self.random_state.randint(2)] = self.random_state.randint(1, donor.size)
        bounds2 = [self.random_state.choice(donor.size + 1 - bounds[1] + bounds[0])]
        bounds2.append(bounds2[0] + bounds[1] - bounds[0])
        assert bounds[1] < donor.size + 1
        donor[bounds[0] : bounds[1]] = individual[bounds2[0] : bounds2[1]]

    def voronoi(self, donor: np.ndarray, individual: np.ndarray) -> None:
        shape = self.shape
        if shape is None or len(shape) < 2:
            warnings.warn("Voronoi DE needs a shape.")
            self.twopoints(donor, individual)
            return
        local_donor = donor.reshape(shape)
        local_individual = individual.reshape(shape)
        x1 = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        x2 = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        x3 = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        x4 = np.array([np.random.randint(shape[i]) for i in range(len(shape))])
        it = np.nditer(local_donor, flags=["multi_index"])
        for _ in it:
            d1 = np.linalg.norm(np.array(it.multi_index) - x1)
            d2 = np.linalg.norm(np.array(it.multi_index) - x2)
            d3 = np.linalg.norm(np.array(it.multi_index) - x3)
            d4 = np.linalg.norm(np.array(it.multi_index) - x4)
            if min([d1, d2, d3]) > d4:
                local_donor[it.multi_index] = local_individual[it.multi_index]
        donor[:] = local_donor.flatten()[:]
        individual[:] = local_individual.flatten()[:]


class _DE(base.Optimizer):
    """Differential evolution.

    Default pop size equal to 30
    We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
    CR =.5, F1=.8, F2=.8, curr-to-best.
    Initial population: pure random.
    """

    # pylint: disable=too-many-locals, too-many-nested-blocks,too-many-instance-attributes
    # pylint: disable=too-many-branches, too-many-statements, too-many-arguments

    def __init__(
        self,
        parametrization: base.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["DifferentialEvolution"] = None,
        weights: tp.Any = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        # config
        self.objective_weights = weights
        self._config = DifferentialEvolution() if config is None else config
        self.scale = (
            float(1.0 / np.sqrt(self.dimension))
            if isinstance(self._config.scale, str)
            else self._config.scale
        )
        pop_choice = {
            "standard": 0,
            "small": 1 + int(np.sqrt(np.log(self.dimension + 3))),
            "dimension": self.dimension + 1,
            "large": 7 * self.dimension,
        }
        if isinstance(self._config.popsize, int):
            self.llambda = self._config.popsize
        else:
            self.llambda = max(30, self.num_workers, pop_choice[self._config.popsize])
        # internals
        # if budget is not None and budget < 60:
        #    warnings.warn(
        #        "DE algorithms are inefficient with budget < 60", base.errors.InefficientSettingsWarning
        #    )
        self._MULTIOBJECTIVE_AUTO_BOUND = max(self._MULTIOBJECTIVE_AUTO_BOUND, self.llambda)  # type: ignore
        self._penalize_cheap_violations = True
        self._uid_queue = base.utils.UidQueue()
        self.population: tp.Dict[str, p.Parameter] = {}
        self.sampler: tp.Optional[base.Optimizer] = None
        self._no_hypervolume = self._config.multiobjective_adaptation

    def set_objective_weights(self, weights: tp.Any) -> None:
        self.objective_weights = weights

    def recommend(self) -> p.Parameter:  # This is NOT the naive version. We deal with noise.
        sample_size = int((self.dimension * (self.dimension - 1)) / 2 + 2 * self.dimension + 1)
        if self._config.high_speed and len(self.archive) >= sample_size:
            try:
                meta_data = metamodel.learn_on_k_best(self.archive, sample_size)
                return self.parametrization.spawn_child().set_standardized_data(meta_data)
            except metamodel.MetaModelFailure:  # The optimum is at infinity. Shit happens.
                pass  # MetaModel failures are something which happens, no worries.
        if self._config.recommendation != "noisy":
            return self.current_bests[self._config.recommendation].parameter
        med_fitness = np.median([p.loss for p in self.population.values() if p.loss is not None])
        good_guys = [p for p in self.population.values() if p.loss is not None and p.loss < med_fitness]
        if not good_guys:
            return self.current_bests["pessimistic"].parameter
        data: tp.Any = sum(
            [g.get_standardized_data(reference=self.parametrization) for g in good_guys]
        ) / len(good_guys)
        out = self.parametrization.spawn_child()
        with p.helpers.deterministic_sampling(out):
            out.set_standardized_data(data)
        return out

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.population) < self.llambda:  # initialization phase
            init = self._config.initialization
            if self.sampler is None and init == "QO":
                self.sampler = oneshot.SamplingSearch(
                    sampler="Hammersley", scrambled=True, opposition_mode="quasi"
                )(self.parametrization, budget=self.llambda)
            if self.sampler is None and init == "SO":
                self.sampler = oneshot.SamplingSearch(
                    sampler="Hammersley", scrambled=True, opposition_mode="special"
                )(self.parametrization, budget=self.llambda)
            if self.sampler is None and init not in ["gaussian", "parametrization"]:
                assert init in ["LHS", "QR"]
                self.sampler = oneshot.SamplingSearch(
                    sampler=init if init == "LHS" else "Hammersley", scrambled=init == "QR", scale=self.scale
                )(
                    self.parametrization,
                    budget=self.llambda,
                )
            if init == "parametrization":
                candidate = self.parametrization.sample()
            elif self.sampler is not None:
                candidate = self.sampler.ask()
            elif self._config.crossover == "voronoi":
                new_guy = (
                    self.scale * self._rng.normal(0, 1, self.dimension)
                    if len(self.population) > self.llambda / 6
                    else self.scale * self._rng.normal() * np.ones(self.dimension)
                )
                candidate = self.parametrization.spawn_child().set_standardized_data(new_guy)
            else:
                new_guy = self.scale * self._rng.normal(0, 1, self.dimension)
                candidate = self.parametrization.spawn_child().set_standardized_data(new_guy)
            candidate.heritage["lineage"] = candidate.uid  # new lineage
            self.population[candidate.uid] = candidate
            self._uid_queue.asked.add(candidate.uid)
            return candidate
        # init is done
        lineage = self._uid_queue.ask()
        parent = self.population[lineage]
        candidate = parent.spawn_child()
        candidate.heritage["lineage"] = lineage  # tell-not-asked may have provided a different lineage
        data = candidate.get_standardized_data(reference=self.parametrization)
        # define all the different parents
        uids = list(self.population)
        a, b = (self.population[uids[self._rng.randint(self.llambda)]] for _ in range(2))
        best = self.current_bests["pessimistic"].parameter
        # redefine the different parents in case of multiobjective optimization
        if self._config.multiobjective_adaptation and self.num_objectives > 1:
            pareto = self.pareto_front()
            # can't use choice directly on pareto, because parametrization can be iterable
            if pareto:
                best = parent if parent in pareto else pareto[self._rng.choice(len(pareto))]
            if len(pareto) > 2:  # otherwise, not enough diversity
                a, b = (pareto[idx] for idx in self._rng.choice(len(pareto), size=2, replace=False))
        # define donor
        data_a, data_b, data_best = (
            indiv.get_standardized_data(reference=self.parametrization) for indiv in (a, b, best)
        )
        donor = data + self._config.F1 * (data_a - data_b) + self._config.F2 * (data_best - data)
        candidate.parents_uids.extend([i.uid for i in (a, b)])
        # apply crossover
        co = self._config.crossover
        if co == "parametrization":
            candidate.recombine(self.parametrization.spawn_child().set_standardized_data(donor))
        else:
            crossovers = Crossover(
                self._rng, 1.0 / self.dimension if co == "dimension" else co, self.parametrization
            )
            crossovers.apply(donor, data)
            candidate.set_standardized_data(donor, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        uid = candidate.heritage["lineage"]
        if uid not in self.population:  # parent was removed, revert to tell_not_asked
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(uid)  # only add to queue if not a "tell_not_asked" (from a removed parent)
        parent = self.population[uid]
        mo_adapt = self._config.multiobjective_adaptation and self.num_objectives > 1
        if mo_adapt:
            if self.objective_weights is None:
                self.objective_weights = np.ones(self.num_objectives)
            else:
                assert len(self.objective_weights) == self.num_objectives
        mo_adapt &= candidate._losses is not None  # can happen with bad constraints
        if not mo_adapt and loss <= base._loss(parent):
            self.population[uid] = candidate
        elif mo_adapt and (
            parent._losses is None
            or np.average(candidate.losses < parent.losses, weights=self.objective_weights) > self._rng.rand()
        ):
            # multiobjective case, with adaptation,
            # randomly replaces the parent depending on the number of better losses
            self.population[uid] = candidate
        elif self._config.propagate_heritage and loss <= float("inf"):
            self.population[uid].heritage.update(candidate.heritage)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        discardable: tp.Optional[str] = None
        if len(self.population) >= self.llambda:

            if self.num_objectives == 1:  # singleobjective: replace if better
                uid, worst = max(self.population.items(), key=lambda p: base._loss(p[1]))
                if loss < base._loss(worst):
                    discardable = uid
            else:  # multiobjective: replace if in pareto and some parents are not
                pareto_uids = {c.uid for c in self.pareto_front()}
                if candidate.uid in pareto_uids:
                    non_pareto_pop = {c.uid for c in self.population.values()} - pareto_uids
                    if non_pareto_pop:
                        nonpareto = {c.uid: c for c in self.population.values()}[list(non_pareto_pop)[0]]
                        discardable = nonpareto.heritage["lineage"]
        if discardable is not None:  # if we found a point to kick, kick it
            del self.population[discardable]
            self._uid_queue.discard(discardable)
        if len(self.population) < self.llambda:  # if there is space, add the new point
            self.population[candidate.uid] = candidate
            # this candidate lineage is not candidate.uid, but to avoid interfering with other optimizers (eg: PSO)
            # we should not update the lineage (and lineage of children must therefore be enforced manually)
            self._uid_queue.tell(candidate.uid)


# pylint: disable=too-many-arguments, too-many-instance-attributes
class DifferentialEvolution(base.ConfiguredOptimizer):
    """Differential evolution is typically used for continuous optimization.
    It uses differences between points in the population for doing mutations in fruitful directions;
    it is therefore a kind of covariance adaptation without any explicit covariance,
    making it super fast in high dimension. This class implements several variants of differential
    evolution, some of them adapted to genetic mutations as in
    `Holland’s work <https://en.wikipedia.org/wiki/Crossover_(genetic_algorithm)#Two-point_and_k-point_crossover>`_),
    (this combination is termed :code:`TwoPointsDE` in Nevergrad, corresponding to :code:`crossover="twopoints"`),
    or to the noisy setting (coined :code:`NoisyDE`, corresponding to :code:`recommendation="noisy"`).
    In that last case, the optimizer returns the mean of the individuals with fitness better than median,
    which might be stupid sometimes though.

    Default settings are CR =.5, F1=.8, F2=.8, curr-to-best, pop size is 30
    Initial population: pure random.

    Parameters
    ----------
    initialization: "parametrization", "LHS" or "QR" or "QO" or "SO"
        algorithm/distribution used for the initialization phase. If "parametrization", this uses the
        sample method of the parametrization.
    scale: float or str
        scale of random component of the updates
    recommendation: "pessimistic", "optimistic", "mean" or "noisy"
        choice of the criterion for the best point to recommend
    crossover: float or str
        crossover rate value, or strategy among:
        - "dimension": crossover rate of  1 / dimension,
        - "random": different random (uniform) crossover rate at each iteration
        - "onepoint": one point crossover
        - "twopoints": two points crossover
        - "rotated_twopoints": more genetic 2p cross-over
        - "parametrization": use the parametrization recombine method
    F1: float
        differential weight #1
    F2: float
        differential weight #2
    popsize: int, "standard", "dimension", "large"
        size of the population to use. "standard" is max(num_workers, 30), "dimension" max(num_workers, 30, dimension +1)
        and "large" max(num_workers, 30, 7 * dimension).
    multiobjective_adaptation: bool
        Automatically adapts to handle multiobjective case.  This is a very basic **experimental** version,
        activated by default because the non-multiobjective implementation is performing very badly.
    high_speed: bool
        Trying to make the optimization faster by a metamodel for the recommendation step.
    """

    def __init__(
        self,
        *,
        initialization: str = "parametrization",
        scale: tp.Union[str, float] = 1.0,
        recommendation: str = "optimistic",
        crossover: tp.Union[str, float] = 0.5,
        F1: float = 0.8,
        F2: float = 0.8,
        popsize: tp.Union[str, int] = "standard",
        propagate_heritage: bool = False,  # experimental
        multiobjective_adaptation: bool = True,
        high_speed: bool = False,
    ) -> None:
        super().__init__(_DE, locals(), as_config=True)
        assert recommendation in ["optimistic", "pessimistic", "noisy", "mean"]
        assert initialization in ["gaussian", "LHS", "QO", "SO", "QR", "parametrization"]

        assert isinstance(scale, float) or scale == "mini"
        if not isinstance(popsize, int):
            assert popsize in ["large", "dimension", "standard", "small"]
        assert isinstance(crossover, float) or crossover in [
            "onepoint",
            "twopoints",
            "rotated_twopoints",
            "dimension",
            "random",
            "parametrization",
            "voronoi",
        ]
        self.initialization = initialization
        self.scale = scale
        self.high_speed = high_speed
        self.recommendation = recommendation
        self.propagate_heritage = propagate_heritage
        self.F1 = F1
        self.F2 = F2
        self.crossover = crossover
        self.popsize = popsize
        self.multiobjective_adaptation = multiobjective_adaptation


DE = DifferentialEvolution().set_name("DE", register=True)
TwoPointsDE = DifferentialEvolution(crossover="twopoints").set_name("TwoPointsDE", register=True)
VoronoiDE = DifferentialEvolution(crossover="voronoi").set_name("VoronoiDE", register=True)
RotatedTwoPointsDE = DifferentialEvolution(crossover="rotated_twopoints").set_name(
    "RotatedTwoPointsDE", register=True
)

LhsDE = DifferentialEvolution(initialization="LHS").set_name("LhsDE", register=True)
QrDE = DifferentialEvolution(initialization="QR").set_name("QrDE", register=True)
QODE = DifferentialEvolution(initialization="QO").set_name("QODE", register=True)
SPQODE = DifferentialEvolution(initialization="QO", popsize="small").set_name("SPQODE", register=True)
QOTPDE = DifferentialEvolution(initialization="QO", crossover="twopoints").set_name("QOTPDE", register=True)
LQOTPDE = DifferentialEvolution(initialization="QO", scale=10.0, crossover="twopoints").set_name(
    "LQOTPDE", register=True
)
LQODE = DifferentialEvolution(initialization="QO", scale=10.0).set_name("LQODE", register=True)
SODE = DifferentialEvolution(initialization="SO").set_name("SODE", register=True)
NoisyDE = DifferentialEvolution(recommendation="noisy").set_name("NoisyDE", register=True)
AlmostRotationInvariantDE = DifferentialEvolution(crossover=0.9).set_name(
    "AlmostRotationInvariantDE", register=True
)
RotationInvariantDE = DifferentialEvolution(crossover=1.0, popsize="dimension").set_name(
    "RotationInvariantDE", register=True
)

# Excellent for hyperparameter tuning.
DiscreteDE = DifferentialEvolution(crossover="dimension").set_name("DiscreteDE", register=True)
