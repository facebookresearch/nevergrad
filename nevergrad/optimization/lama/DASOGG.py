import numpy as np


class DASOGG:
    def __init__(self, budget, population_size=50, spiral_rate=0.6, beta=0.3, gradient_descent_factor=0.05):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.spiral_rate = spiral_rate
        self.beta = beta
        self.gradient_descent_factor = gradient_descent_factor

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
                r1, r2, r3 = np.random.rand(3)  # Random coefficients
                # Dynamic spiral updating rule with gradient guidance
                dynamic_adjustment = self.beta * np.tanh(num_evals / self.budget)
                local_gradient = best_individual - population[i]
                global_gradient = np.random.uniform(self.lb, self.ub, self.dimension) - population[i]

                velocities[i] = (
                    r1 * velocities[i]
                    + r2 * self.spiral_rate * local_gradient
                    + dynamic_adjustment * global_gradient
                    + r3 * self.gradient_descent_factor * local_gradient
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
