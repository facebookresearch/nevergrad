import numpy as np


class OptimallyBalancedQuantumStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_ratio=0.2,
        mutation_scale_base=0.5,
        mutation_decay=0.05,
        crossover_rate=0.8,
        quantum_update_rate=0.9,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_ratio)
        self.mutation_scale_base = mutation_scale_base
        self.mutation_decay = mutation_decay
        self.crossover_rate = crossover_rate
        self.quantum_update_rate = quantum_update_rate

    def __call__(self, func):
        # Initialize population randomly within the search space
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Select elites
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.random() < self.crossover_rate:
                    parent1, parent2 = np.random.choice(elite_indices, 2, replace=False)
                    offspring = self.crossover(population[parent1], population[parent2])
                else:
                    offspring = population[np.random.choice(elite_indices)]

                if np.random.random() < self.quantum_update_rate:
                    offspring = self.quantum_state_update(offspring, best_individual)

                mutation_scale = self.adaptive_mutation_scale(evaluations)
                offspring += np.random.normal(0, mutation_scale, self.dimension)
                offspring = np.clip(offspring, -5, 5)

                new_population[i] = offspring

            # Evaluate new population
            fitness = np.array([func(x) for x in new_population])
            evaluations += self.population_size

            # Update best individual
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = new_population[current_best_idx]

            population = new_population

        return best_fitness, best_individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_state_update(self, individual, best_individual):
        perturbation = np.random.normal(0, 0.1, self.dimension)
        return best_individual + perturbation * (best_individual - individual)

    def adaptive_mutation_scale(self, evaluations):
        return self.mutation_scale_base * np.exp(-self.mutation_decay * evaluations / self.budget)
