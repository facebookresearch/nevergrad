import numpy as np


class RefinedAdaptiveHybridOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Given dimensionality.
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 200
        mutation_factor = 0.8  # Increased initial mutation to explore more broadly
        crossover_prob = 0.7  # Lowered crossover to prevent premature convergence
        adaptive_factor = 0.01  # Smoother adaptation

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        num_iterations = self.budget // population_size

        for iteration in range(num_iterations):
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                # Dynamic mutation factor adjustment
                dynamic_mutation = mutation_factor + adaptive_factor * np.random.randn()
                mutant = a + dynamic_mutation * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Dynamic crossover probability adjustment
                dynamic_crossover = crossover_prob + adaptive_factor * np.random.randn()
                trial = np.where(np.random.rand(self.dim) < dynamic_crossover, mutant, population[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial.copy()

            # Adaptive mutation and crossover adjustments
            if best_value < np.mean(fitness):
                mutation_factor += 2 * adaptive_factor  # Faster increase in mutation factor
                crossover_prob -= adaptive_factor / 2  # Slower decrease in crossover probability
            else:
                mutation_factor -= adaptive_factor / 2  # Slower decrease in mutation factor
                crossover_prob += 2 * adaptive_factor  # Faster increase in crossover probability

            mutation_factor = np.clip(mutation_factor, 0.1, 1.0)
            crossover_prob = np.clip(crossover_prob, 0.1, 1.0)

        return best_value, best_solution
