import numpy as np


class AdaptiveDiversifiedSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        # Initialize solution and function value tracking
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population_size = 50
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        # Initial best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        # Main optimization loop
        for iteration in range(self.budget):
            for i in range(population_size):
                # Mutation strategy: Adaptive perturbation
                perturbation_scale = 0.5 * (1 - iteration / self.budget)  # Decreases over time
                perturbation = np.random.normal(0, perturbation_scale, self.dim)
                candidate = population[i] + perturbation

                # Ensure candidate stays within bounds
                candidate = np.clip(candidate, self.lb, self.ub)

                # Evaluate candidate
                candidate_fitness = func(candidate)

                # Acceptance condition: Greedy selection
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness

                    # Update the global best solution
                    if candidate_fitness < self.f_opt:
                        self.f_opt = candidate_fitness
                        self.x_opt = candidate.copy()

        return self.f_opt, self.x_opt


# Example of use (requires a function `func` and bounds setup to run):
# optimizer = AdaptiveDiversifiedSearch(budget=10000)
# best_value, best_solution = optimizer(func)
