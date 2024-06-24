import numpy as np


class AdaptiveQuantumCrossoverOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        current_budget = 0
        population_size = 200  # Increased population size for more diversity
        mutation_factor = 0.9  # Initial mutation factor
        crossover_prob = 0.8  # Initial crossover probability
        adaptivity_rate = 0.05  # Rate at which parameters adapt

        # Initialize population and fitness
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while current_budget < self.budget:
            new_population = np.empty_like(population)

            # Elite-based reproduction with adaptation
            elite_size = int(population_size * 0.1)  # Top 10% as elite
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_population = population[elite_indices]

            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Select parents from elite
                parents_indices = np.random.choice(elite_indices, 2, replace=False)
                parent1, parent2 = population[parents_indices]

                # Crossover
                mask = np.random.rand(self.dim) < crossover_prob
                child = np.where(mask, parent1, parent2)

                # Quantum-Inspired mutation based on Gaussian noise
                quantum_noise = np.random.randn(self.dim) * mutation_factor
                child += quantum_noise
                child = np.clip(child, self.lower_bound, self.upper_bound)

                child_fitness = func(child)
                current_budget += 1

                # Update the best solution if found
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

                new_population[i] = child

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adaptive mechanism for mutation and crossover, responding to landscape
            mutation_factor -= mutation_factor * adaptivity_rate
            crossover_prob += (1 - crossover_prob) * adaptivity_rate

        return best_fitness, best_solution
