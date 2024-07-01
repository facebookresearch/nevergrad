import numpy as np


class QuantumEnhancedDynamicHybridSearchV9:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=300,
        elite_ratio=0.15,
        mutation_scale=0.4,
        mutation_decay=0.002,
        crossover_prob=0.95,
        quantum_intensity=0.3,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(np.floor(population_size * elite_ratio))
        self.mutation_scale = mutation_scale
        self.mutation_decay = mutation_decay
        self.crossover_prob = crossover_prob
        self.quantum_intensity = quantum_intensity

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Track the best individual
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            for i in range(self.population_size):
                if np.random.random() < self.crossover_prob:
                    parent1 = population[np.random.choice(elite_indices)]
                    parent2 = population[np.random.choice(elite_indices)]
                    child = self.crossover(parent1, parent2)
                else:
                    child = population[np.random.choice(elite_indices)]

                if self.quantum_intensity and np.random.random() < 0.25:
                    child = self.quantum_tuning(child, best_individual)

                mutation_scale_adjusted = self.mutation_scale * np.exp(-self.mutation_decay * evaluations)
                child += np.random.normal(0, mutation_scale_adjusted, self.dimension)
                child = np.clip(child, -5, 5)

                new_population[i] = child

            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            best_new_idx = np.argmin(new_fitness)
            if new_fitness[best_new_idx] < best_fitness:
                best_fitness = new_fitness[best_new_idx]
                best_individual = new_population[best_new_idx]

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            sorted_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[sorted_indices]
            fitness = combined_fitness[sorted_indices]

        return best_fitness, best_individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dimension)
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_tuning(self, individual, best_individual):
        perturbation = np.random.uniform(-1, 1, self.dimension) * 0.1  # Increased uniform perturbation scale
        return individual + perturbation * (best_individual - individual)
