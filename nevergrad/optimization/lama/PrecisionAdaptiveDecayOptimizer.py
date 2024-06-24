import numpy as np


class PrecisionAdaptiveDecayOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the optimization problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimization setup
        current_budget = 0
        population_size = 150
        num_elites = 20  # Increased elite population for better preservation
        mutation_factor = 0.85  # Lower initial mutation factor
        crossover_rate = 0.7  # Higher initial crossover rate for better exploration

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Optimization loop
        while current_budget < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Elite preservation
            elite_indices = np.argsort(fitness)[:num_elites]
            new_population[:num_elites] = population[elite_indices]
            new_fitness[:num_elites] = fitness[elite_indices]

            # Generate new solutions
            for i in range(num_elites, population_size):
                if current_budget >= self.budget:
                    break

                # Differential mutation based on random selection
                indices = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Binomial crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate new solution
                trial_fitness = func(trial)
                current_budget += 1

                # Selection step
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

                # Update the best found solution
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness

            # Update the population and fitness
            population = new_population
            fitness = new_fitness

            # Adapt mutation factor and crossover rate dynamically
            mutation_factor *= 0.98  # Slower decay of mutation factor
            crossover_rate = min(0.95, crossover_rate * 1.01)  # Gradual increase of crossover rate

        return best_fitness, best_solution
