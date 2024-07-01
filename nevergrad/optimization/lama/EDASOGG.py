import numpy as np


class EDASOGG:
    def __init__(
        self,
        budget,
        population_size=30,
        spiral_rate=0.5,
        beta=0.2,
        gradient_descent_factor=0.07,
        momentum=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.spiral_rate = spiral_rate  # Adjusted for more controlled movement
        self.beta = beta  # Lowered to reduce rapid changes in dynamics
        self.gradient_descent_factor = (
            gradient_descent_factor  # Slightly increased for better local exploitation
        )
        self.momentum = momentum  # New parameter to incorporate historical velocity influence

    def __call__(self, func):
        # Initialize population, velocities, and personal bests
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        velocities = np.zeros((self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        pbest = population.copy()
        pbest_fitness = fitness.copy()

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        # Evolutionary loop
        while num_evals < self.budget:
            for i in range(self.population_size):
                r1, r2, r3, r4 = np.random.rand(4)  # Random coefficients

                # Enhanced velocity update formula with momentum and dynamic adjustments
                velocities[i] = (
                    self.momentum * velocities[i]
                    + r1 * self.gradient_descent_factor * (pbest[i] - population[i])
                    + r2 * self.spiral_rate * (best_individual - population[i])
                    + r3 * self.beta * (np.random.uniform(self.lb, self.ub, self.dimension) - population[i])
                )

                # Update position
                population[i] += velocities[i]
                population[i] = np.clip(population[i], self.lb, self.ub)

                # Evaluate
                updated_fitness = func(population[i])
                num_evals += 1

                # Update personal and global best
                if updated_fitness < pbest_fitness[i]:
                    pbest[i] = population[i]
                    pbest_fitness[i] = updated_fitness

                if updated_fitness < best_fitness:
                    best_fitness = updated_fitness
                    best_individual = population[i]

                if num_evals >= self.budget:
                    break

        return best_fitness, best_individual
