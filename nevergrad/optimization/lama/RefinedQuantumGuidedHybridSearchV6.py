import numpy as np


class RefinedQuantumGuidedHybridSearchV6:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=150,
        elite_ratio=0.2,
        mutation_scale=0.5,
        mutation_decay=0.005,
        crossover_prob=0.8,
        quantum_factor=0.9,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(np.ceil(population_size * elite_ratio))
        self.mutation_scale = mutation_scale
        self.mutation_decay = mutation_decay
        self.crossover_prob = crossover_prob
        self.quantum_factor = quantum_factor

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
            # Generate a new population
            new_population = np.empty_like(population)
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            for i in range(self.population_size):
                if np.random.random() < self.crossover_prob:
                    # Crossover from elite individuals
                    parent1 = population[np.random.choice(elite_indices)]
                    parent2 = population[np.random.choice(elite_indices)]
                    child = self.crossover(parent1, parent2)
                else:
                    # Directly copy an elite with a probability
                    child = population[np.random.choice(elite_indices)]

                # Apply quantum tuning on a probability
                if np.random.random() < self.quantum_factor:
                    child = self.quantum_tuning(child, best_individual)

                # Mutation with decreasing scale
                mutation_scale = self.mutation_scale * np.exp(
                    -self.mutation_decay * evaluations / self.budget
                )
                child += np.random.normal(0, mutation_scale, self.dimension)
                child = np.clip(child, -5, 5)

                new_population[i] = child

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Select the best from the new population
            best_new_idx = np.argmin(new_fitness)
            if new_fitness[best_new_idx] < best_fitness:
                best_fitness = new_fitness[best_new_idx]
                best_individual = new_population[best_new_idx]

            # Combine and sort populations based on fitness
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
        perturbation = np.random.normal(-0.1, 0.1, self.dimension)
        return individual + perturbation * (best_individual - individual)
