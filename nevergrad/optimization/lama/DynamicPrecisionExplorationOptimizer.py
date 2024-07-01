import numpy as np


class DynamicPrecisionExplorationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension fixed as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 50  # Adjusted population for broader exploration
        mutation_rate = 0.1  # Initial mutation rate
        exploration_depth = 0.05  # Depth of exploration around the current best
        crossover_probability = 0.7  # Probability of crossover

        # Initialize population within the bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            indices = np.arange(population_size)
            np.random.shuffle(indices)
            for i in indices:
                if np.random.rand() < crossover_probability:
                    # Crossover operation between two random individuals and the best solution
                    idx_a, idx_b = np.random.choice(population_size, 2, replace=False)
                    a, b = population[idx_a], population[idx_b]
                    mutant = a + mutation_rate * (best_solution - b)
                else:
                    # Mutation only operation for more extensive exploration
                    mutant = population[i] + np.random.normal(0, exploration_depth, self.dim)

                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial_fitness = func(mutant)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = mutant
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = mutant

            # Dynamic adjustments based on the progress
            if evaluations % (self.budget // 20) == 0:  # Adjustments at finer intervals
                mutation_rate *= 0.95  # Decrease mutation rate for finer search as time progresses
                exploration_depth *= 0.9  # Reduce exploration depth to focus on local optima

        return best_fitness, best_solution
