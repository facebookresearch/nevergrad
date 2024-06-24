import numpy as np


class EnhancedSpatialAdaptiveOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimization setup
        current_budget = 0
        population_size = 100
        mutation_factor = 0.6  # Initiate with a moderate mutation to balance exploration
        crossover_prob = 0.7  # Moderate crossover probability to maintain diversity

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Begin main optimization loop
        while current_budget < self.budget:
            new_population = np.empty_like(population)
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Select three random individuals different from i
                indices = np.arange(population_size)
                indices = np.delete(indices, i)
                x1, x2, x3 = population[np.random.choice(indices, 3, replace=False)]

                # Mutation using current-to-pbest/1 strategy
                p_best_index = np.argmin(fitness)
                p_best = population[p_best_index]
                mutant = x1 + mutation_factor * (p_best - x1 + x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_prob
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                current_budget += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                else:
                    new_population[i] = population[i]

            population = new_population

            # Adaptive parameter tuning based on performance
            performance_feedback = np.std(fitness) / np.mean(fitness) if np.mean(fitness) != 0 else 0
            mutation_factor = max(0.1, min(0.9, mutation_factor + 0.1 * (0.2 - performance_feedback)))
            crossover_prob = max(0.5, min(0.9, crossover_prob + 0.1 * (0.1 - performance_feedback)))

        return best_fitness, best_solution
