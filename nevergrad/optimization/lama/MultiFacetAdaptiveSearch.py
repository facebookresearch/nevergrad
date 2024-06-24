import numpy as np


class MultiFacetAdaptiveSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        # Initialize solution and function value tracking
        self.f_opt = np.Inf
        self.x_opt = None

        # Dynamic population scaling and random restarts
        initial_population_size = 50
        population_size = initial_population_size
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        # Initial best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        # Main optimization loop
        evaluations_used = population_size
        while evaluations_used < self.budget:
            # Adaptive mutation based on remaining budget
            remaining_budget = self.budget - evaluations_used
            mutation_scale = 0.1 * (remaining_budget / self.budget) + 0.02

            # Evolve population
            for i in range(population_size):
                perturbation = np.random.normal(0, mutation_scale, self.dim)
                candidate = population[i] + perturbation
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                evaluations_used += 1

                # Accept if better or with a probability decreasing over time
                if candidate_fitness < fitness[i] or np.random.rand() < (
                    0.5 * remaining_budget / self.budget
                ):
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < self.f_opt:
                        self.f_opt = candidate_fitness
                        self.x_opt = candidate.copy()

            # Random restart mechanism
            if evaluations_used + population_size <= self.budget:
                if np.random.rand() < 0.1:  # 10% chance of random restart
                    new_individuals = np.random.uniform(self.lb, self.ub, (population_size // 2, self.dim))
                    new_fitness = np.array([func(individual) for individual in new_individuals])
                    # Replace the worst half of the population
                    worst_half_indices = np.argsort(fitness)[-population_size // 2 :]
                    population[worst_half_indices] = new_individuals
                    fitness[worst_half_indices] = new_fitness
                    evaluations_used += population_size // 2

        return self.f_opt, self.x_opt


# Example of usage (requires a function `func` and bounds to run):
# optimizer = MultiFacetAdaptiveSearch(budget=10000)
# best_value, best_solution = optimizer(func)
