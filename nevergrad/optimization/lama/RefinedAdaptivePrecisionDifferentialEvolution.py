import numpy as np


class RefinedAdaptivePrecisionDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The given dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = (
            100  # Adjusted population size for balance between performance and computational cost
        )
        mutation_factor = 0.9  # Initial high mutation factor to promote diverse exploration
        crossover_prob = 0.85  # Starting crossover probability
        adaptive_threshold = 0.2  # Modified threshold for adapting mutation and crossover

        # Initialize population randomly
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        # Main optimization loop
        for generation in range(self.budget // population_size):
            # Adapt mutation factor and crossover probability based on progress and fitness improvements
            progress = generation / (self.budget // population_size)
            if progress > adaptive_threshold and progress < 0.5:
                mutation_factor *= 0.98  # Gradually decrease mutation factor
                crossover_prob *= 0.99  # Gradually decrease crossover probability
            elif progress >= 0.5:
                mutation_factor *= 0.97  # Further decrease mutation factor for fine-tuning
                crossover_prob *= 0.97  # Further decrease for stable convergence

            for i in range(population_size):
                # Mutation using "current-to-best/1/bin" strategy for faster convergence on promising solutions
                indices = [j for j in range(population_size) if j != i]
                a, b = population[np.random.choice(indices, 2, replace=False)]
                best = population[best_idx]
                mutant = population[i] + mutation_factor * (best - population[i]) + mutation_factor * (a - b)
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
