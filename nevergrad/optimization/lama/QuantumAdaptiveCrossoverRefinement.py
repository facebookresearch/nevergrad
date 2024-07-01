import numpy as np


class QuantumAdaptiveCrossoverRefinement:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 400  # Further increased population size for diversity
        mutation_factor = 0.85  # More aggressive initial mutation factor
        crossover_prob = 0.75  # Higher initial crossover probability
        adaptivity_rate = 0.05  # Slower change rate for better stability

        # Initialize population and fitness
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while current_budget < self.budget:
            new_population = np.empty_like(population)

            # Refinement: Increase elite size to 30% for better quality samples
            elite_size = int(population_size * 0.3)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Selecting parents adaptively with preference to elites
                if np.random.rand() < 0.7:  # Higher chance to select from elite
                    parents_indices = np.random.choice(elite_indices, 2, replace=False)
                    parent1, parent2 = population[parents_indices]
                else:
                    parent1, parent2 = population[np.random.choice(range(population_size), 2, replace=False)]

                # Crossover
                mask = np.random.rand(self.dim) < crossover_prob
                child = np.where(mask, parent1, parent2)

                # Mutation with dynamic adaptivity based on trigonometric modulation
                dynamic_mutation = mutation_factor * (1 + np.sin(2 * np.pi * current_budget / self.budget))
                quantum_noise = np.random.randn(self.dim) * dynamic_mutation
                child += quantum_noise
                child = np.clip(child, self.lower_bound, self.upper_bound)

                child_fitness = func(child)
                current_budget += 1

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adaptive mutation and crossover adjustments using a slower rate
            mutation_factor *= 1 - adaptivity_rate
            crossover_prob = np.clip(crossover_prob + adaptivity_rate * (np.random.rand() - 0.5), 0.5, 1)

        return best_fitness, best_solution
