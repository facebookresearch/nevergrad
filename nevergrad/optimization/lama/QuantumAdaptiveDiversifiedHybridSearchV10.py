import numpy as np


class QuantumAdaptiveDiversifiedHybridSearchV10:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=350,
        elite_ratio=0.2,
        mutation_scale=0.5,
        mutation_decay=0.0015,
        crossover_prob=0.8,
        quantum_intensity=0.35,
        local_search_prob=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(np.floor(population_size * elite_ratio))
        self.mutation_scale = mutation_scale
        self.mutation_decay = mutation_decay
        self.crossover_prob = crossover_prob
        self.quantum_intensity = quantum_intensity
        self.local_search_prob = local_search_prob

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

                if np.random.random() < self.local_search_prob:
                    child = self.local_search(child, func)

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

    def local_search(self, individual, func):
        local_step_size = 0.1
        for _ in range(int(self.dimension / 2)):  # Limited local search steps
            perturbation = np.random.uniform(-local_step_size, local_step_size, self.dimension)
            new_individual = individual + perturbation
            new_individual = np.clip(new_individual, -5, 5)
            if func(new_individual) < func(individual):
                individual = new_individual
        return individual
