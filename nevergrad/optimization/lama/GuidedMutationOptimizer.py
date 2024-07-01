import numpy as np


class GuidedMutationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimization setup
        current_budget = 0
        population_size = 200  # Further increased population size for enhanced exploration
        mutation_factor = 0.8  # Initial mutation factor set higher to promote diverse searching
        crossover_prob = 0.7  # Initial crossover probability

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Main optimization loop
        while current_budget < self.budget:
            new_population = np.empty_like(population)
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Use guided mutation based on the best solution
                indices = np.arange(population_size)
                indices = np.delete(indices, i)
                x1, x2, x3 = population[np.random.choice(indices, 3, replace=False)]

                # Mutation: guided by best and random selection
                mutant = best_solution + mutation_factor * (x1 - best_solution + x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])

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
            best_index = np.argmin(fitness)

            # Adaptive parameters adjustment
            mutation_factor = max(0.1, mutation_factor - 0.02)  # Gradually decrease mutation factor
            crossover_prob = min(0.9, crossover_prob + 0.02)  # Gradually increase crossover probability

        return best_fitness, best_solution
