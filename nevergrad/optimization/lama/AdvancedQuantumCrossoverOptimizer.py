import numpy as np


class AdvancedQuantumCrossoverOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initial setup
        current_budget = 0
        population_size = 100  # Increased population for broader initial sampling
        mutation_factor = 0.5  # Reduced mutation factor for stability
        crossover_prob = 0.5  # Moderated crossover probability for maintaining diversity
        elite_factor = 0.1  # Fraction of population considered elite

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

            # Generate new population with quantum-inspired crossover
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Selection from elite pool for quantum crossover
                parent1 = elite_population[np.random.randint(0, elite_size)]
                parent2 = elite_population[np.random.randint(0, elite_size)]
                child = np.where(np.random.rand(self.dim) < crossover_prob, parent1, parent2)

                # Mutation inspired by quantum tunneling effect
                quantum_mutation = mutation_factor * np.random.randn(self.dim)
                child += quantum_mutation
                child = np.clip(child, self.lower_bound, self.upper_bound)

                child_fitness = func(child)
                current_budget += 1

                # Maintain the best solution found
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Dynamic adaptation of mutation factor and crossover probability
            mutation_factor *= 0.99  # Gradual decrease to assure convergence
            crossover_prob *= 1.01  # Incremental increase to keep exploring new areas

        return best_fitness, best_solution
