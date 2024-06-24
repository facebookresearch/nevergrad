import numpy as np


class RefinedAdaptiveExplorationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Search space dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 300  # Increased population size for better exploration
        mutation_base = 0.5  # Lower base mutation factor for finer grained exploration
        mutation_adaptiveness = 0.1  # Adaptiveness factor for mutation control
        crossover_base = 0.8  # Base crossover rate
        elite_size = 30  # Increased elite size for better retention of solutions

        # Initialize population and evaluate
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            # Elite retention
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[elite_indices]
            new_fitness[:elite_size] = fitness[elite_indices]

            for i in range(elite_size, population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]

                # Adaptive mutation with noise reduction as evaluations increase
                noise_reduction_factor = 1 - (evaluations / self.budget)
                mutation_factor = (
                    mutation_base + mutation_adaptiveness * (np.random.rand() - 0.5) * noise_reduction_factor
                )
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_base
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

                # Update best solution
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

            population = new_population
            fitness = new_fitness

        return best_fitness, best_solution
