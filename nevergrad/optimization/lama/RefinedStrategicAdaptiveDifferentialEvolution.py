import numpy as np


class RefinedStrategicAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality set to 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 200  # Increased population size for more diverse solutions
        mutation_factor = 0.5  # Start with a lower mutation factor for detailed initial exploration
        crossover_prob = 0.9  # Higher crossover probability to promote better mixing of solutions
        adaptive_factor = 0.98  # Slower reduction in mutation and crossover to maintain exploration

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        # Adaptation thresholds
        no_improve_threshold = 0.1 * self.budget / population_size  # More frequent adaptation checks

        for generation in range(int(self.budget / population_size)):
            if generation % no_improve_threshold == 0 and generation != 0:
                # Increase mutation and crossover if no improvement
                mutation_factor = min(1.2 * mutation_factor, 1.0)
                crossover_prob = min(1.1 * crossover_prob, 1.0)

            # Evolution strategy
            for i in range(population_size):
                indices = [j for j in range(population_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover (binomial)
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

                    # Update best solution
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial.copy()

            # Adaptive mutation and crossover decrease
            mutation_factor *= adaptive_factor
            crossover_prob *= adaptive_factor

        return best_value, best_solution
