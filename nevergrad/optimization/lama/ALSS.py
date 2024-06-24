import numpy as np


class ALSS:
    def __init__(self, budget, population_size=50, learning_rate=0.6):
        self.budget = budget
        self.population_size = population_size
        self.learning_rate = learning_rate  # Learning rate for step size adjustment
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        # Initialize step sizes for each individual in each dimension
        step_sizes = np.random.uniform(0.1, 1.0, (self.population_size, self.dimension))

        while num_evals < self.budget:
            for i in range(self.population_size):
                # Perturb each individual based on its step size
                perturbation = np.random.normal(0, step_sizes[i], self.dimension)
                trial_vector = population[i] + perturbation
                trial_vector = np.clip(trial_vector, self.lb, self.ub)

                # Fitness evaluation
                trial_fitness = func(trial_vector)
                num_evals += 1

                # Adaptive adjustment
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    step_sizes[i] *= 1 + self.learning_rate  # Increase step size
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector
                else:
                    step_sizes[i] *= 1 - self.learning_rate  # Decrease step size

                if num_evals >= self.budget:
                    break

        return best_fitness, best_individual


# Usage of ALSS:
# optimizer = ALSS(budget=1000)
# best_fitness, best_solution = optimizer(func)
