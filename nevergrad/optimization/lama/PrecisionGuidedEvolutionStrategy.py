import numpy as np


class PrecisionGuidedEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        # Initialize parameters
        population_size = 50
        children_multiplier = 7  # Number of children per parent
        mutation_strength = 0.5  # Initial mutation strength
        success_threshold = 0.2  # Threshold for successful mutations

        # Create initial population and evaluate it
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        # Track the best solution found
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        evaluations = population_size
        successful_mutations = 0
        attempted_mutations = 0

        while evaluations < self.budget:
            new_population = []
            new_fitness = []

            for parent in population:
                for _ in range(children_multiplier):
                    child = parent + np.random.normal(0, mutation_strength, self.dim)
                    child = np.clip(child, self.lb, self.ub)
                    child_fitness = func(child)

                    new_population.append(child)
                    new_fitness.append(child_fitness)
                    evaluations += 1

                    attempted_mutations += 1
                    if child_fitness < func(parent):
                        successful_mutations += 1

                    if evaluations >= self.budget:
                        break
                if evaluations >= self.budget:
                    break

            # Update the population with the best performing individuals
            total_population = np.vstack((population, new_population))
            total_fitness = np.hstack((fitness, new_fitness))
            best_indices = np.argsort(total_fitness)[:population_size]
            population = total_population[best_indices]
            fitness = total_fitness[best_indices]

            # Update the best found solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx].copy()

            # Adapt mutation strength
            if attempted_mutations > 0:
                success_ratio = successful_mutations / attempted_mutations
                if success_ratio > success_threshold:
                    mutation_strength /= 0.85  # Increase mutation strength
                else:
                    mutation_strength *= 0.85  # Decrease mutation strength

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = PrecisionGuidedEvolutionStrategy(budget=10000)
# best_value, best_solution = optimizer(func)
