import numpy as np
from scipy.optimize import minimize


class EnhancedAdaptiveDynamicMultiStrategyDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 100
        self.F = 0.8
        self.CR = 0.9
        self.local_search_prob = 0.3
        self.restart_threshold = 50
        self.strategy_weights = np.ones(4)
        self.strategy_success = np.zeros(4)
        self.learning_rate = 0.1
        self.no_improvement_count = 0
        self.elite_fraction = 0.2
        self.history = []
        self.dynamic_adjustment_period = 20
        self.dynamic_parameters_adjustment_threshold = 30
        self.pop_shrink_factor = 0.1

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def _local_search(self, x, func):
        bounds = [(self.lb, self.ub)] * self.dim
        result = minimize(func, x, method="L-BFGS-B", bounds=bounds)
        return result.x, result.fun

    def _dynamic_parameters(self):
        self.F = np.clip(self.F + np.random.normal(0, self.learning_rate), 0.5, 1.5)
        self.CR = np.clip(self.CR + np.random.normal(0, self.learning_rate), 0.2, 1.0)

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

    def __call__(self, func):
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        self.evaluations = len(population)

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        while self.evaluations < self.budget:
            new_population = []
            new_fitness = []

            for i in range(self.pop_size):
                if self.evaluations >= self.budget:
                    break

                strategy = self._select_strategy()
                indices = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3, r4, r5 = np.random.choice(indices, 5, replace=False)
                best_idx = np.argmin(fitness)

                if strategy == self._mutation_best_1:
                    donor = self._mutation_best_1(population, best_idx, r1, r2)
                elif strategy == self._mutation_rand_1:
                    donor = self._mutation_rand_1(population, r1, r2, r3)
                elif strategy == self._mutation_rand_2:
                    donor = self._mutation_rand_2(population, r1, r2, r3, r4, r5)
                else:  # strategy == self._mutation_best_2
                    donor = self._mutation_best_2(population, best_idx, r1, r2, r3, r4)

                trial = np.clip(donor, self.lb, self.ub)
                if np.random.rand() < self.CR:
                    trial = np.where(np.random.rand(self.dim) < self.CR, trial, population[i])
                else:
                    trial = population[i]

                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < fitness[i]:
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
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            if np.random.rand() < self.local_search_prob:
                elite_indices = np.argsort(fitness)[: int(self.elite_fraction * self.pop_size)]
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

            if self.no_improvement_count >= self.restart_threshold:
                population = self._initialize_population()
                fitness = np.array([func(ind) for ind in population])
                self.evaluations += len(population)
                self.no_improvement_count = 0

            if self.no_improvement_count >= self.dynamic_parameters_adjustment_threshold:
                self.strategy_weights = (self.strategy_success + 1) / (self.strategy_success.sum() + 4)
                self.strategy_success.fill(0)
                self.no_improvement_count = 0
                self._dynamic_parameters()

            # Dynamic population resizing based on performance
            if self.no_improvement_count >= self.dynamic_adjustment_period:
                new_pop_size = max(20, int(self.pop_size * (1 - self.pop_shrink_factor)))
                population = population[:new_pop_size]
                fitness = fitness[:new_pop_size]
                self.pop_size = new_pop_size
                self.no_improvement_count = 0

            self.history.append(self.f_opt)

        return self.f_opt, self.x_opt
