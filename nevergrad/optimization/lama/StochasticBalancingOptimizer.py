import numpy as np


class StochasticBalancingOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the optimization problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.epsilon = 1e-8  # To prevent division by zero in adaptive mechanisms

    def __call__(self, func):
        # Optimization setup
        current_budget = 0
        population_size = 100
        mutation_factor = 0.5  # Dynamic mutation factor initialization
        crossover_prob = 0.7  # Crossover probability

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while current_budget < self.budget:
            new_population = np.empty_like(population)
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                inds = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[inds]

                # Mutation step
                mutant = x1 + mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover step
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < crossover_prob else population[i][j]
                        for j in range(self.dim)
                    ]
                )

                trial_fitness = func(trial)
                current_budget += 1

                # Selection step
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                else:
                    new_population[i] = population[i]

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            current_budget += population_size

            # Adapt mutation and crossover rates based on diversity and fitness improvements
            diversity = np.std(population)
            if diversity < 1e-1:  # Low diversity triggers more exploration
                mutation_factor = min(1.0, mutation_factor + 0.1)
            else:
                mutation_factor = max(0.1, mutation_factor - 0.02)

            crossover_prob = min(
                1.0,
                crossover_prob
                + 0.05 * (1 - diversity / (self.upper_bound - self.lower_bound + self.epsilon)),
            )

        return best_fitness, best_solution
