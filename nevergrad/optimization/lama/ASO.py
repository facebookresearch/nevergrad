import numpy as np


class ASO:
    def __init__(self, budget, population_size=100, spiral_rate=0.5):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.spiral_rate = spiral_rate

    def __call__(self, func):
        # Initialize population and velocities
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        velocities = np.zeros((self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        # Evolutionary loop
        while num_evals < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)  # Random coefficients
                # Spiral updating rule
                velocities[i] = (
                    r1 * velocities[i]
                    + r2 * self.spiral_rate * (best_individual - population[i])
                    + self.spiral_rate * (np.random.uniform(self.lb, self.ub, self.dimension) - population[i])
                )

                # Update position
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lb, self.ub)

                # Evaluate
                updated_fitness = func(population[i])
                num_evals += 1

                # Selection
                if updated_fitness < fitness[i]:
                    fitness[i] = updated_fitness
                    if updated_fitness < best_fitness:
                        best_fitness = updated_fitness
                        best_individual = population[i]

                if num_evals >= self.budget:
                    break

        return best_fitness, best_individual
