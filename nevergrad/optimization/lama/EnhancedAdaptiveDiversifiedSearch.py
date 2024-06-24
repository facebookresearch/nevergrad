import numpy as np


class EnhancedAdaptiveDiversifiedSearch:
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
        population_size = 100
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        # Initial best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        # Main optimization loop
        num_iterations = int(self.budget / population_size)
        for iteration in range(num_iterations):
            for i in range(population_size):
                # Mutation strategy: Adaptive perturbation with occasional large jumps
                if np.random.rand() < 0.1:  # 10% chance for a larger mutation
                    perturbation_scale = 1.0 - (
                        iteration / num_iterations
                    )  # Larger mutation at the beginning
                else:
                    perturbation_scale = 0.1 * (1 - iteration / num_iterations)  # Standard mutation scale

                perturbation = np.random.normal(0, perturbation_scale, self.dim)
                candidate = population[i] + perturbation

                # Ensure candidate stays within bounds
                candidate = np.clip(candidate, self.lb, self.ub)

                # Evaluate candidate
                candidate_fitness = func(candidate)

                # Acceptance condition: Greedy selection with elitism
                if (
                    candidate_fitness < fitness[i] or np.random.rand() < 0.05
                ):  # 5% chance to accept worse solutions
                    population[i] = candidate
                    fitness[i] = candidate_fitness

                    # Update the global best solution
                    if candidate_fitness < self.f_opt:
                        self.f_opt = candidate_fitness
                        self.x_opt = candidate.copy()

        return self.f_opt, self.x_opt


# Example of use (requires a function `func` and bounds to run):
# optimizer = EnhancedAdaptiveDiversifiedSearch(budget=10000)
# best_value, best_solution = optimizer(func)
