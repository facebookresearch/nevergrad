import numpy as np
from scipy.optimize import minimize


class DynamicAdaptivePopulationDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0
        self.initial_pop_size = 100
        self.min_pop_size = 20
        self.num_subpopulations = 5
        self.subpop_size = self.initial_pop_size // self.num_subpopulations
        self.strategy_weights = np.ones(4)
        self.strategy_success = np.zeros(4)
        self.learning_rate = 0.1
        self.no_improvement_count = 0
        self.F = 0.5
        self.CR = 0.9

    def _initialize_population(self, pop_size):
        return np.random.uniform(self.lb, self.ub, (pop_size, self.dim))

    def _local_search(self, x, func):
        res = minimize(func, x, method="L-BFGS-B", bounds=[(self.lb, self.ub)] * self.dim)
        return res.x, res.fun

    def _dynamic_parameters(self):
        self.F = np.clip(self.F + np.random.normal(0, self.learning_rate), 0.1, 1.0)
        self.CR = np.clip(self.CR + np.random.normal(0, self.learning_rate), 0.1, 1.0)

    def _mutation_best_1(self, population, best_idx, r1, r2):
        return population[best_idx] + self.F * (population[r1] - population[r2])

    def _mutation_rand_1(self, population, r1, r2, r3):
        return population[r1] + self.F * (population[r2] - population[r3])

    def _mutation_rand_2(self, population, r1, r2, r3, r4, r5):
        return (
            population[r1]
            + self.F * (population[r2] - population[r3])
            + self.F * (population[r4] - population[r5])
        )

    def _mutation_best_2(self, population, best_idx, r1, r2, r3, r4):
        return (
            population[best_idx]
            + self.F * (population[r1] - population[r2])
            + self.F * (population[r3] - population[r4])
        )

    def _select_strategy(self):
        return np.random.choice(
            [self._mutation_best_1, self._mutation_rand_1, self._mutation_rand_2, self._mutation_best_2],
            p=self.strategy_weights / self.strategy_weights.sum(),
        )

    def _opposition_based_learning(self, population):
        return self.lb + self.ub - population

    def _crowding_distance(self, population, fitness):
        distances = np.zeros(len(population))
        sorted_indices = np.argsort(fitness)
        for i in range(self.dim):
            sorted_pop = population[sorted_indices, i]
            distances[sorted_indices[0]] = distances[sorted_indices[-1]] = float("inf")
            for j in range(1, len(population) - 1):
                distances[sorted_indices[j]] += (sorted_pop[j + 1] - sorted_pop[j - 1]) / (
                    sorted_pop[-1] - sorted_pop[0] + 1e-12
                )
        return distances

    def __call__(self, func):
        population = self._initialize_population(self.initial_pop_size)
        fitness = np.array([func(ind) for ind in population])
        self.evaluations = len(population)

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        while self.evaluations < self.budget:
            subpopulations = np.array_split(population, self.num_subpopulations)
            subfitness = np.array_split(fitness, self.num_subpopulations)
            new_population = []
            new_fitness = []

            for subpop, subfit in zip(subpopulations, subfitness):
                for i in range(len(subpop)):
                    if self.evaluations >= self.budget:
                        break

                    strategy = self._select_strategy()
                    indices = [idx for idx in range(len(subpop)) if idx != i]
                    r1, r2, r3, r4, r5 = np.random.choice(indices, 5, replace=False)
                    best_idx = np.argmin(subfit)

                    if strategy == self._mutation_best_1:
                        donor = self._mutation_best_1(subpop, best_idx, r1, r2)
                    elif strategy == self._mutation_rand_1:
                        donor = self._mutation_rand_1(subpop, r1, r2, r3)
                    elif strategy == self._mutation_rand_2:
                        donor = self._mutation_rand_2(subpop, r1, r2, r3, r4, r5)
                    else:  # strategy == self._mutation_best_2
                        donor = self._mutation_best_2(subpop, best_idx, r1, r2, r3, r4)

                    trial = np.clip(donor, self.lb, self.ub)
                    f_trial = func(trial)
                    self.evaluations += 1

                    if f_trial < subfit[i]:
                        new_population.append(trial)
                        new_fitness.append(f_trial)
                        strategy_idx = [
                            self._mutation_best_1,
                            self._mutation_rand_1,
                            self._mutation_rand_2,
                            self._mutation_best_2,
                        ].index(strategy)
                        self.strategy_success[strategy_idx] += 1
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial
                            self.no_improvement_count = 0
                    else:
                        new_population.append(subpop[i])
                        new_fitness.append(subfit[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            # Perform local search on elite solutions
            elite_indices = np.argsort(fitness)[: self.num_subpopulations]
            for idx in elite_indices:
                if self.evaluations >= self.budget:
                    break
                x_local, f_local = self._local_search(population[idx], func)
                self.evaluations += 1
                if f_local < fitness[idx]:
                    population[idx] = x_local
                    fitness[idx] = f_local
                    if f_local < self.f_opt:
                        self.f_opt = f_local
                        self.x_opt = x_local
                        self.no_improvement_count = 0

            if self.no_improvement_count >= 5:
                population = self._initialize_population(self.initial_pop_size)
                fitness = np.array([func(ind) for ind in population])
                self.evaluations += len(population)
                self.no_improvement_count = 0

            # Crowding distance to maintain diversity
            distances = self._crowding_distance(population, fitness)
            sorted_indices = np.argsort(distances)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]

            # Opposition-based learning
            if self.evaluations < self.budget:
                opp_population = self._opposition_based_learning(population)
                opp_fitness = np.array([func(ind) for ind in opp_population])
                self.evaluations += len(opp_population)
                combined_population = np.concatenate((population, opp_population), axis=0)
                combined_fitness = np.concatenate((fitness, opp_fitness), axis=0)
                sorted_indices = np.argsort(combined_fitness)[: self.initial_pop_size]
                population = combined_population[sorted_indices]
                fitness = combined_fitness[sorted_indices]

            self.strategy_weights = self.strategy_success + 1
            self.strategy_success.fill(0)
            self.no_improvement_count += 1

            # Dynamic adjustment of population size
            if self.no_improvement_count >= 10:
                reduced_pop_size = max(self.min_pop_size, len(population) - 10)
                population = population[:reduced_pop_size]
                fitness = fitness[:reduced_pop_size]
                self.subpop_size = len(population) // self.num_subpopulations
                self.no_improvement_count = 0

            self._dynamic_parameters()

        return self.f_opt, self.x_opt
