import numpy as np


class RAGEA:
    def __init__(self, budget, population_size=50, alpha=0.1, mutation_scaling=0.1, elite_fraction=0.2):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = alpha  # Adaptive step scaling factor
        self.mutation_scaling = mutation_scaling
        self.elite_count = int(population_size * elite_fraction)

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            sorted_indices = np.argsort(fitness)
            elites = population[sorted_indices[: self.elite_count]]
            elite_fitnesses = fitness[sorted_indices[: self.elite_count]]

            new_population = np.zeros_like(population)
            new_population[: self.elite_count] = elites

            # Generate new candidates
            for i in range(self.elite_count, self.population_size):
                parent_idx = np.random.choice(self.elite_count)
                parent = elites[parent_idx]

                # Adaptive mutation based on elite performance
                relative_fitness = (elite_fitnesses[parent_idx] - np.min(elite_fitnesses)) / (
                    np.max(elite_fitnesses) - np.min(elite_fitnesses) + 1e-8
                )
                step_size = self.mutation_scaling * (1 - relative_fitness)
                mutation = np.random.normal(0, max(step_size, 1e-8), self.dimension)

                # Adaptive crossover between best individual and elite
                crossover_weight = np.random.rand()
                child = (crossover_weight * best_individual) + ((1 - crossover_weight) * parent) + mutation
                child = np.clip(child, self.lb, self.ub)  # Ensure bounds are respected

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            num_evals += self.population_size - self.elite_count

            # Update the best individual if necessary
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx].copy()

        return best_fitness, best_individual
