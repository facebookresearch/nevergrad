import numpy as np


class EnhancedAdaptiveGuidedMutationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimization setup
        current_budget = 0
        population_size = 300  # Slightly increased population size for better exploration
        mutation_factor = (
            0.8  # Adjusted initial mutation factor to balance exploration and robust convergence
        )
        crossover_prob = 0.7  # Adjusted initial crossover probability for more frequent trials

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Enhance with Local Search Initialization
        local_search_frequency = 200  # frequency of localized search
        local_search_radius = 0.1  # radius of the local search

        # Main optimization loop
        while current_budget < self.budget:
            new_population = np.empty_like(population)
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Mutation and crossover phases
                indices = np.arange(population_size)
                indices = np.delete(indices, i)
                random_indices = np.random.choice(indices, 3, replace=False)
                x1, x2, x3 = population[random_indices]

                # Hybrid mutation strategy with local search tweak
                if current_budget % local_search_frequency == 0:
                    # Conduct a local search around the current best
                    local_mutant = best_solution + local_search_radius * np.random.randn(self.dim)
                    local_mutant = np.clip(local_mutant, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_mutant)
                    current_budget += 1

                    if local_fitness < best_fitness:
                        best_solution = local_mutant
                        best_fitness = local_fitness

                mutant = population[i] + mutation_factor * (
                    best_solution - population[i] + x1 - (x2 + x3) / 2
                )
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

            # Adaptively adjust mutation and crossover parameters
            mutation_factor = max(0.5, mutation_factor - 0.005)  # Slower decrease
            crossover_prob = min(0.9, crossover_prob + 0.005)  # Slower increase

        return best_fitness, best_solution
