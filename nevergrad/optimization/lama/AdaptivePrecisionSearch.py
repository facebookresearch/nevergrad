import numpy as np


class AdaptivePrecisionSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration parameters
        population_size = 100
        mutation_factor = 0.5
        crossover_rate = 0.9
        elite_size = max(1, int(population_size * 0.1))

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Evolutionary loop
        for _ in range(int(self.budget / population_size)):
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Keep a portion of the best solutions (elitism)
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[elite_indices]
            new_fitness[:elite_size] = fitness[elite_indices]

            # Generate new candidates
            for i in range(elite_size, population_size):
                # Mutation (differential evolution strategy)
                idxs = np.random.choice(population_size, 3, replace=False)
                x0, x1, x2 = population[idxs]

                mutant = x0 + mutation_factor * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

            # Replace old population
            population = new_population
            fitness = new_fitness

            # Update the best solution found
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_solution = population[current_best_index]

        return best_fitness, best_solution
