import numpy as np


class EACDE:
    def __init__(
        self, budget, population_size=50, F_base=0.5, CR_base=0.9, cluster_ratio=0.25, adaptation_frequency=50
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.cluster_ratio = cluster_ratio  # Ratio of population to consider for clustering
        self.adaptation_frequency = adaptation_frequency  # Frequency of adaptation for F and CR

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main evolutionary loop
        while num_evals < self.budget:
            # Clustering top-performing individuals
            top_cluster_size = int(self.population_size * self.cluster_ratio)
            top_indices = np.argsort(fitness)[:top_cluster_size]
            top_cluster = population[top_indices]

            # Adaptive strategy parameters based on performance
            if num_evals % self.adaptation_frequency == 0:
                successful_indices = fitness < np.median(fitness)
                if np.any(successful_indices):
                    self.F_base = np.clip(np.mean(fitness[successful_indices]), 0.1, 1)
                    self.CR_base = np.clip(np.mean(fitness[successful_indices]), 0.5, 1)

            for i in range(self.population_size):
                if np.random.rand() < np.mean(fitness) / best_fitness:
                    # Higher mutation rate near best individual
                    F = self.F_base + 0.1 * np.random.randn()
                else:
                    F = self.F_base

                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)

                mutant = population[a] + F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                CR = self.CR_base + 0.1 * np.random.randn()
                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial vector
                trial_fitness = func(trial)
                num_evals += 1
                if num_evals >= self.budget:
                    break

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

        return best_fitness, best_individual
