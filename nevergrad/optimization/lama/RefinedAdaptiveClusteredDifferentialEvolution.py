import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol


class RefinedAdaptiveClusteredDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem statement
        self.lb = -5.0
        self.ub = 5.0
        self.pop_size = 100
        self.memory_size = 20
        self.elite_size = 5
        self.memory = []
        self.elite = []
        self.mutation_strategies = [self._mutation_best_1, self._mutation_rand_1, self._mutation_rand_2]
        self.strategy_weights = np.ones(len(self.mutation_strategies))
        self._dynamic_parameters()

    def _initialize_population(self):
        sobol_engine = Sobol(d=self.dim, scramble=False)
        sobol_samples = sobol_engine.random_base2(m=int(np.log2(self.pop_size // 2)))
        sobol_samples = self.lb + (self.ub - self.lb) * sobol_samples

        random_samples = np.random.uniform(self.lb, self.ub, (self.pop_size - len(sobol_samples), self.dim))
        return np.vstack((sobol_samples, random_samples))

    def _local_search(self, x, func):
        res = minimize(
            func, x, method="L-BFGS-B", bounds=[(self.lb, self.ub)] * self.dim, options={"disp": False}
        )
        return res.x, res.fun

    def _dynamic_parameters(self):
        self.F = np.random.uniform(0.5, 1.0)
        self.CR = np.random.uniform(0.4, 0.9)

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

    def _select_strategy(self):
        return np.random.choice(
            self.mutation_strategies, p=self.strategy_weights / self.strategy_weights.sum()
        )

    def _opposition_based_learning(self, population):
        opp_population = self.lb + self.ub - population
        return opp_population

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
                else:  # strategy == self._mutation_rand_2
                    donor = self._mutation_rand_2(population, r1, r2, r3, r4, r5)

                trial = np.clip(donor, self.lb, self.ub)
                f_trial = func(trial)
                self.evaluations += 1

                if f_trial < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(f_trial)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])

            population = np.array(new_population)
            fitness = np.array(new_fitness)

            elite_indices = np.argsort(fitness)[: self.elite_size]
            self.elite = [population[idx] for idx in elite_indices]

            if self.evaluations < self.budget:
                if len(self.memory) < self.memory_size:
                    self.memory.append(self.x_opt)
                else:
                    worst_mem_idx = np.argmax([func(mem) for mem in self.memory])
                    self.memory[worst_mem_idx] = self.x_opt

            self._dynamic_parameters()

            if self.evaluations < self.budget:
                opp_population = self._opposition_based_learning(population)
                opp_fitness = np.array([func(ind) for ind in opp_population])
                self.evaluations += len(opp_population)
                population = np.concatenate((population, opp_population), axis=0)
                fitness = np.concatenate((fitness, opp_fitness), axis=0)
                sorted_indices = np.argsort(fitness)[: self.pop_size]
                population = population[sorted_indices]
                fitness = fitness[sorted_indices]

        return self.f_opt, self.x_opt
