import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans


class MultiPopulationAdaptiveMemorySearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        # Parameters for DE
        self.population_size = 100
        self.num_sub_populations = 5
        self.F = 0.8
        self.CR = 0.9

        # PSO Parameters
        self.inertia_weight = 0.9
        self.cognitive_constant = 2.0
        self.social_constant = 2.0

        # Memory Mechanism
        self.memory_size = 20
        self.memory = []

    def _initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))

    def _nelder_mead_local_search(self, x, func):
        res = minimize(func, x, method="nelder-mead", options={"xtol": 1e-8, "disp": False})
        return res.x, res.fun

    def _adaptive_parameter_adjustment(self):
        self.F = np.random.uniform(0.4, 1.0)
        self.CR = np.random.uniform(0.1, 1.0)
        self.inertia_weight = np.random.uniform(0.4, 0.9)

    def _cluster_based_search(self, population, fitness, func):
        if len(population) > 10:
            kmeans = KMeans(n_clusters=10).fit(population)
            cluster_centers = kmeans.cluster_centers_
            for center in cluster_centers:
                local_candidate, f_local_candidate = self._nelder_mead_local_search(center, func)
                self.evaluations += 1
                if f_local_candidate < self.f_opt:
                    self.f_opt = f_local_candidate
                    self.x_opt = local_candidate

    def _memory_based_search(self, func):
        if len(self.memory) > 1:
            for mem in self.memory:
                local_candidate, f_local_candidate = self._nelder_mead_local_search(mem, func)
                self.evaluations += 1
                if f_local_candidate < self.f_opt:
                    self.f_opt = f_local_candidate
                    self.x_opt = local_candidate

    def __call__(self, func):
        # Initialize population
        population = self._initialize_population()
        fitness = np.array([func(ind) for ind in population])
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)].copy()

        personal_best_positions = population.copy()
        personal_best_fitness = fitness.copy()

        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            # Divide population into subpopulations
            sub_pop_size = self.population_size // self.num_sub_populations
            sub_populations = [
                population[i * sub_pop_size : (i + 1) * sub_pop_size] for i in range(self.num_sub_populations)
            ]
            sub_fitness = [
                fitness[i * sub_pop_size : (i + 1) * sub_pop_size] for i in range(self.num_sub_populations)
            ]

            # Perform DE in subpopulations
            for sub_pop, sub_fit in zip(sub_populations, sub_fitness):
                for i in range(len(sub_pop)):
                    # Select three random vectors a, b, c from subpopulation
                    indices = [idx for idx in range(len(sub_pop)) if idx != i]
                    a, b, c = sub_pop[np.random.choice(indices, 3, replace=False)]

                    # Mutation and Crossover
                    mutant_vector = np.clip(a + self.F * (b - c), self.lb, self.ub)
                    trial_vector = np.copy(sub_pop[i])
                    for j in range(self.dim):
                        if np.random.rand() < self.CR:
                            trial_vector[j] = mutant_vector[j]

                    f_candidate = func(trial_vector)
                    self.evaluations += 1

                    if f_candidate < sub_fit[i]:
                        sub_pop[i] = trial_vector
                        sub_fit[i] = f_candidate

                        if f_candidate < self.f_opt:
                            self.f_opt = f_candidate
                            self.x_opt = trial_vector

                    if self.evaluations >= self.budget:
                        break

            # Recombine subpopulations
            population = np.vstack(sub_populations)
            fitness = np.hstack(sub_fitness)

            # PSO component
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocities = (
                self.inertia_weight * velocities
                + self.cognitive_constant * r1 * (personal_best_positions - population)
                + self.social_constant * r2 * (self.x_opt - population)
            )
            population = np.clip(population + velocities, self.lb, self.ub)

            # Evaluate new population
            for i in range(self.population_size):
                f_candidate = func(population[i])
                self.evaluations += 1

                if f_candidate < fitness[i]:
                    fitness[i] = f_candidate
                    personal_best_positions[i] = population[i]
                    personal_best_fitness[i] = f_candidate

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = population[i]

                if self.evaluations >= self.budget:
                    break

            # Memory mechanism
            if len(self.memory) < self.memory_size:
                self.memory.append(self.x_opt)
            else:
                worst_mem_idx = np.argmax([func(mem) for mem in self.memory])
                self.memory[worst_mem_idx] = self.x_opt

            # Adaptive Parameter Adjustment
            self._adaptive_parameter_adjustment()

            # Cluster-Based Enhanced Local Search
            self._cluster_based_search(population, fitness, func)

            # Memory-Based Search
            self._memory_based_search(func)

        return self.f_opt, self.x_opt
