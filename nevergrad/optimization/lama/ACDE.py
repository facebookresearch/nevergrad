import numpy as np


class ACDE:
    def __init__(self, budget, population_size=100, F=0.8, CR=0.9, cluster_ratio=0.2, adaptation_rate=0.05):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.cluster_ratio = cluster_ratio  # Ratio of population to consider for clustering
        self.adaptation_rate = adaptation_rate  # Rate of adaptation for F and CR

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

            # Update strategy parameters adaptively
            if num_evals > self.population_size and num_evals % 100 == 0:
                self.F += self.adaptation_rate * (1 - 2 * np.random.rand())
                self.CR += self.adaptation_rate * (1 - 2 * np.random.rand())
                self.F, self.CR = np.clip(self.F, 0.5, 1), np.clip(self.CR, 0.8, 1)

            for i in range(self.population_size):
                if i in top_indices:
                    # Evolve using cluster members
                    a, b, c = np.random.choice(top_indices, 3, replace=False)
                else:
                    # Regular DE operation
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(indices, 3, replace=False)

                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.CR
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
