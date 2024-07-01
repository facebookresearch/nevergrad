import numpy as np


class QuantumInfluenceCrossoverOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 150  # Increase population size for broader sampling
        mutation_factor = 0.8  # Adjusted mutation factor for controlled exploration
        crossover_prob = 0.7  # Higher crossover probability for better information exchange
        elite_factor = 0.2  # Increased elite fraction for enhanced quality propagation

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while current_budget < self.budget:
            new_population = np.empty_like(population)
            elite_size = int(population_size * elite_factor)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Enhanced selection strategy
                parent1 = elite_population[np.random.randint(0, elite_size)]
                parent2 = elite_population[np.random.randint(0, elite_size)]
                child = np.where(np.random.rand(self.dim) < crossover_prob, parent1, parent2)

                # Quantum-inspired mutation
                quantum_mutation = mutation_factor * (
                    np.random.randn(self.dim) * (1 - np.exp(-np.random.rand(self.dim)))
                )
                child += quantum_mutation
                child = np.clip(child, self.lower_bound, self.upper_bound)

                child_fitness = func(child)
                current_budget += 1

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Dynamic adaptation adjustments
            mutation_factor *= 0.95  # Gradual decrease
            crossover_prob *= 1.02  # Gradual increase to prevent premature convergence

        return best_fitness, best_solution
