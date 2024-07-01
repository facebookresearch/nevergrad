import numpy as np


class DDPO:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])

    def initialize(self):
        population_size = 40
        population = np.random.uniform(*self.bounds, (population_size, self.dimension))
        return population, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def dual_search(self, population, func):
        global_best_fitness = np.Inf
        global_best_individual = None
        local_best_fitness = np.Inf
        local_best_individual = None

        evaluations = 0
        while evaluations < self.budget:
            fitness = self.evaluate(population, func)
            evaluations += len(population)

            # Update best solutions
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < global_best_fitness:
                global_best_fitness = fitness[best_idx]
                global_best_individual = population[best_idx]

            # Global exploration
            exploration_population = np.random.uniform(*self.bounds, (len(population) // 2, self.dimension))
            exploration_fitness = self.evaluate(exploration_population, func)
            evaluations += len(exploration_population)

            # Local exploitation
            perturbations = np.random.normal(0, 0.1, (len(population) // 2, self.dimension))
            exploitation_population = population[: len(population) // 2] + perturbations
            exploitation_population = np.clip(exploitation_population, *self.bounds)
            exploitation_fitness = self.evaluate(exploitation_population, func)
            evaluations += len(exploitation_population)

            # Combine and select
            combined_population = np.vstack([exploration_population, exploitation_population])
            combined_fitness = np.concatenate([exploration_fitness, exploitation_fitness])
            if np.min(combined_fitness) < local_best_fitness:
                local_best_fitness = np.min(combined_fitness)
                local_best_individual = combined_population[np.argmin(combined_fitness)]

            # Feedback-driven dynamic adjustments
            if local_best_fitness < global_best_fitness * 1.05:  # Detection of potential local optimum
                perturbations = np.random.normal(
                    0, 0.5, (len(population), self.dimension)
                )  # Enhanced exploration
                population = global_best_individual + perturbations
                population = np.clip(population, *self.bounds)
            else:
                population = combined_population

        return global_best_fitness, global_best_individual

    def __call__(self, func):
        initial_population, _ = self.initialize()
        best_fitness, best_solution = self.dual_search(initial_population, func)
        return best_fitness, best_solution
