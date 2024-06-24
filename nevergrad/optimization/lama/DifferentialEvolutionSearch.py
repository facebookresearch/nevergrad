import numpy as np


class DifferentialEvolutionSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize parameters
        population_size = 20
        crossover_rate = 0.9
        differential_weight = 0.8

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        self.f_opt = np.min(fitness)
        self.x_opt = population[np.argmin(fitness)]

        evaluations = population_size

        while evaluations < self.budget:
            new_population = np.copy(population)

            for i in range(population_size):
                # Randomly select three indices that are not the current index
                indices = list(range(population_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Generate trial vector
                mutant_vector = np.clip(a + differential_weight * (b - c), self.lb, self.ub)

                crossover_mask = np.random.rand(self.dim) < crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True

                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial_vector

            population = new_population

        return self.f_opt, self.x_opt


# Example usage:
# def sample_func(x):
#     return np.sum(x**2)

# optimizer = DifferentialEvolutionSearch(budget=10000)
# best_fitness, best_solution = optimizer(sample_func)
# print("Best fitness:", best_fitness)
# print("Best solution:", best_solution)
