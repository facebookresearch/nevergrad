import numpy as np


class GEEA:
    def __init__(self, budget, population_size=30, alpha=0.5, beta=0.1, gamma=0.1):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = alpha  # Learning rate for exploration
        self.beta = beta  # Learning rate for exploitation
        self.gamma = gamma  # Mutation factor for diversity

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        while num_evals < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Exploration: Select random individuals
                indices = np.random.permutation(self.population_size)
                x1, x2, x3 = population[indices[:3]]

                # Mutation and crossover
                mutant_vector = x1 + self.gamma * (x2 - x3)
                mutant_vector = np.clip(mutant_vector, self.lb, self.ub)

                # Crossover (binomial)
                trial_vector = np.where(
                    np.random.rand(self.dimension) < self.beta, mutant_vector, population[i]
                )

                # Exploitation: Learning from the best
                trial_vector += self.alpha * (best_individual - population[i])
                trial_vector = np.clip(trial_vector, self.lb, self.ub)

                # Fitness evaluation
                trial_fitness = func(trial_vector)
                num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population.append(trial_vector)
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector
                else:
                    new_population.append(population[i])

                if num_evals >= self.budget:
                    break

            population = np.array(new_population)

        return best_fitness, best_individual


# Usage of GEEA:
# optimizer = GEEA(budget=1000)
# best_fitness, best_solution = optimizer(func)
