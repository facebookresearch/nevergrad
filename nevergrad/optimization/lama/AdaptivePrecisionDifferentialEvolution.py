import numpy as np


class AdaptivePrecisionDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The given dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 150  # Slightly reduced population for faster generations
        mutation_factor = 0.8  # Increased mutation factor for more aggressive exploration
        crossover_prob = 0.9  # High crossover for aggressive recombination
        adaptive_threshold = 0.1  # Threshold for adapting mutation and crossover

        # Initialize population randomly
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        # Main optimization loop
        for generation in range(self.budget // population_size):
            # Adapt mutation factor and crossover probability based on progress
            progress = generation / (self.budget // population_size)
            if progress > adaptive_threshold:
                mutation_factor *= 0.95  # Decrease to fine-tune exploration
                crossover_prob *= 0.98  # Decrease to stabilize gene propagation

            for i in range(population_size):
                # Mutation using "rand/1" strategy
                indices = [j for j in range(population_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover - Binomial
                trial = np.array(
                    [
                        mutant[j] if np.random.rand() < crossover_prob else population[i][j]
                        for j in range(self.dim)
                    ]
                )

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial.copy()

        return best_value, best_solution
