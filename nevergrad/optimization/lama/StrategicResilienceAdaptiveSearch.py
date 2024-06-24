import numpy as np


class StrategicResilienceAdaptiveSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        # Initial solution and function value tracking
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population_size = 100
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        # Find initial best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        # Stagnation counter to enable strategic diversity injections
        stagnation_counter = 0

        # Main optimization loop
        evaluations_used = population_size
        while evaluations_used < self.budget:
            # Adjust mutation scale based on remaining budget and stagnation
            remaining_budget = self.budget - evaluations_used
            mutation_scale = 0.1 * (remaining_budget / self.budget) + 0.05 * (stagnation_counter / 10)

            # Generate new candidates
            for i in range(population_size):
                perturbation = np.random.normal(0, mutation_scale, self.dim)
                candidate = population[i] + perturbation
                candidate = np.clip(candidate, self.lb, self.ub)
                candidate_fitness = func(candidate)
                evaluations_used += 1

                # Solution acceptance or rejection
                if candidate_fitness < fitness[i]:
                    population[i] = candidate
                    fitness[i] = candidate_fitness
                    if candidate_fitness < self.f_opt:
                        self.f_opt = candidate_fitness
                        self.x_opt = candidate.copy()
                        stagnation_counter = 0  # Reset stagnation counter on improvement
                else:
                    stagnation_counter += 1

            # Strategic diversity injection mechanism
            if stagnation_counter >= 50:
                stagnation_counter = 0  # Reset counter
                # Inject new random solutions
                new_inds = np.random.uniform(self.lb, self.ub, (population_size // 4, self.dim))
                new_fitness = np.array([func(ind) for ind in new_inds])
                # Replace a quarter of the worst solutions
                worst_indices = np.argsort(fitness)[-population_size // 4 :]
                population[worst_indices] = new_inds
                fitness[worst_indices] = new_fitness
                evaluations_used += population_size // 4

        return self.f_opt, self.x_opt


# Example of usage (requires a function `func` and bounds to run):
# optimizer = StrategicResilienceAdaptiveSearch(budget=10000)
# best_value, best_solution = optimizer(func)
