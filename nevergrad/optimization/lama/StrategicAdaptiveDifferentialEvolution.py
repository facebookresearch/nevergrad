import numpy as np


class StrategicAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality as given
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 150  # Increased population size for robust exploration of search space
        mutation_factor = 0.8  # Moderately high initial mutation factor for broad exploration
        crossover_prob = 0.7  # Relatively lower crossover probability to maintain diversity
        adaptive_factor = 0.95  # Adaptive factor to modify mutation and crossover based on fitness trend

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        # Analyze and adapt strategy parameters dynamically
        last_improvement = 0  # Track last improvement generation
        improvements = 0  # Track total improvements

        for generation in range(int(self.budget / population_size)):
            if generation - last_improvement > 50:  # If no improvement over 50 generations
                mutation_factor *= 1.05  # Increase mutation factor to escape potential local minima
                crossover_prob *= 1.05  # Increase crossover probability to encourage diversity
                last_improvement = generation  # Reset last improvement tracker

            # Generate new population
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

                # Selection and update
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Check if this is the best solution found
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial.copy()
                        improvements += 1
                        last_improvement = generation

            # Adapt mutation and crossover probabilistically based on historical improvements
            if improvements > 0:
                mutation_factor *= adaptive_factor
                crossover_prob *= adaptive_factor

        return best_value, best_solution
