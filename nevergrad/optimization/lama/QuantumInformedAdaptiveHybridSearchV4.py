import numpy as np


class QuantumInformedAdaptiveHybridSearchV4:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_ratio=0.1,
        mutation_scale=1.0,
        mutation_decay=0.01,
        crossover_prob=0.7,
        quantum_boost=0.95,
        refinement_rate=0.05,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(np.ceil(population_size * elite_ratio))
        self.mutation_scale = mutation_scale
        self.mutation_decay = mutation_decay
        self.crossover_prob = crossover_prob
        self.quantum_boost = quantum_boost
        self.refinement_rate = refinement_rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Perform crossover and mutation
            for i in range(self.population_size):
                if np.random.random() < self.crossover_prob:
                    parents = np.random.choice(elite_indices, 2, replace=False)
                    child = self.crossover(population[parents[0]], population[parents[1]])
                else:
                    child = population[np.random.choice(elite_indices)]

                # Quantum boost decision
                if np.random.random() < self.quantum_boost:
                    child = self.quantum_state_modification(child, best_individual)

                # Mutation with decaying scale
                mutation_scale = self.mutation_scale * np.exp(
                    -self.mutation_decay * evaluations / self.budget
                )
                child += np.random.normal(0, mutation_scale, self.dimension)
                child = np.clip(child, -5, 5)

                new_population[i] = child

            # Evaluate new population
            fitness = np.array([func(x) for x in new_population])
            evaluations += self.population_size

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = new_population[current_best_idx]

            population = new_population

        return best_fitness, best_individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_state_modification(self, individual, best_individual):
        perturbation = np.random.normal(0, self.refinement_rate, self.dimension)
        return individual + perturbation * (best_individual - individual)
