import numpy as np


class AdaptiveQuantumEvolvedDiversityExplorerV15:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.2,
        mutation_intensity=0.8,
        crossover_rate=0.6,
        quantum_prob=0.9,
        gamma=0.9,
        beta=0.1,
        epsilon=0.01,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.quantum_prob = quantum_prob
        self.gamma = gamma  # Quantum state update influence
        self.beta = beta  # Mutation decay rate
        self.epsilon = epsilon  # Minimum mutation factor

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.random() < self.crossover_rate:
                    parent_indices = np.random.choice(elite_indices, 2, replace=False)
                    child = self.crossover(population[parent_indices[0]], population[parent_indices[1]])
                else:
                    parent_idx = np.random.choice(elite_indices)
                    child = self.mutate(population[parent_idx], evaluations)

                if np.random.random() < self.quantum_prob:
                    child = self.quantum_state_update(child, best_individual)

                new_population[i] = np.clip(child, -5, 5)

            population = new_population
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            new_best_idx = np.argmin(fitness)
            if fitness[new_best_idx] < best_fitness:
                best_fitness = fitness[new_best_idx]
                best_individual = population[new_best_idx]

        return best_fitness, best_individual

    def mutate(self, individual, evaluations):
        intensity = max(
            self.epsilon, self.mutation_intensity * np.exp(-self.beta * evaluations / self.budget)
        )
        return individual + np.random.normal(0, intensity, self.dimension)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dimension)
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_state_update(self, individual, best_individual):
        perturbation = np.random.normal(0, self.gamma, self.dimension) * (best_individual - individual)
        return individual + perturbation
