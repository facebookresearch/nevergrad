import numpy as np


class AdaptiveEliteDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        # Parameters
        self.initial_population_size = 100  # More reasonable initial population size
        self.F_min = 0.5
        self.F_max = 0.9
        self.CR_min = 0.7
        self.CR_max = 0.9
        self.local_search_chance = 0.2
        self.elite_ratio = 0.1
        self.diversity_threshold = 0.05
        self.cauchy_step_scale = 0.02
        self.gaussian_step_scale = 0.005
        self.reinitialization_rate = 0.1
        self.hyper_heuristic_probability = 0.5

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.initial_population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        evaluations = self.initial_population_size

        while evaluations < self.budget:
            self.adaptive_parameters_adjustment(evaluations)

            # Sort population based on fitness
            sorted_indices = np.argsort(fitness)
            elite_size = int(self.elite_ratio * len(population))
            elite_population = population[sorted_indices[:elite_size]]

            new_population = []
            for i in range(len(population)):
                if np.random.rand() < self.local_search_chance:
                    candidate = self.local_search(population[i], func)
                else:
                    # Differential Evolution mutation and crossover
                    idxs = np.random.choice(len(population), 3, replace=False)
                    a, b, c = population[idxs]
                    mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                    crossover = np.random.rand(self.dim) < self.CR
                    candidate = np.where(crossover, mutant, population[i])

                # Selection
                f_candidate = func(candidate)
                evaluations += 1
                if f_candidate < fitness[i]:
                    new_population.append(candidate)
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = candidate
                else:
                    new_population.append(population[i])

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

            # Add elite back to population
            population = np.vstack((population, elite_population))
            fitness = np.array([func(ind) for ind in population])
            evaluations += elite_size

            # Adaptive control of parameters based on population diversity
            self.adaptive_population_reinitialization(population, evaluations)

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        best_x = x.copy()
        best_f = func(x)

        for _ in range(20):  # Fewer local search iterations for more efficient exploration
            step_size_cauchy = np.random.standard_cauchy(self.dim) * self.cauchy_step_scale
            step_size_gaussian = np.random.normal(0, self.gaussian_step_scale, size=self.dim)

            x_new_cauchy = np.clip(best_x + step_size_cauchy, self.lb, self.ub)
            x_new_gaussian = np.clip(best_x + step_size_gaussian, self.lb, self.ub)

            f_new_cauchy = func(x_new_cauchy)
            f_new_gaussian = func(x_new_gaussian)

            if f_new_cauchy < best_f:
                best_x = x_new_cauchy
                best_f = f_new_cauchy
            elif f_new_gaussian < best_f:
                best_x = x_new_gaussian
                best_f = f_new_gaussian

        return best_x

    def adaptive_parameters_adjustment(self, evaluations):
        progress_ratio = evaluations / self.budget
        self.F = self.F_min + (self.F_max - self.F_min) * progress_ratio
        self.CR = self.CR_min + (self.CR_max - self.CR_min) * progress_ratio

    def adaptive_population_reinitialization(self, population, evaluations):
        diversity = np.mean(np.std(population, axis=0))

        if diversity < self.diversity_threshold:
            num_reinit = int(self.reinitialization_rate * len(population))
            reinit_indices = np.random.choice(len(population), num_reinit, replace=False)

            for idx in reinit_indices:
                population[idx] = np.random.uniform(self.lb, self.ub, self.dim)

        remaining_budget_ratio = (self.budget - evaluations) / self.budget
        self.local_search_chance = max(0.1, self.local_search_chance * remaining_budget_ratio)
