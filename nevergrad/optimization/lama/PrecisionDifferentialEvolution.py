import numpy as np


class PrecisionDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The given dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 200  # Increased population for greater search diversity
        mutation_factor = 0.6  # Base mutation factor
        crossover_prob = 0.8  # High crossover probability for more robust search

        # Initialize population randomly
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        # Main optimization loop
        for _ in range(self.budget // population_size):
            for i in range(population_size):
                # Mutation
                indices = [j for j in range(population_size) if j != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])

                # Selection: Evaluate the trial solution
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution if the new solution is better
                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial.copy()

        return best_value, best_solution
