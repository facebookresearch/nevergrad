import numpy as np


class EnhancedQuantumAdaptiveHybridSearchV2:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_frac=0.3,
        mutation_intensity=0.8,
        crossover_prob=0.7,
        quantum_prob=0.8,
        gradient_prob=0.6,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_frac)
        self.mutation_intensity = mutation_intensity
        self.crossover_prob = crossover_prob
        self.quantum_prob = quantum_prob
        self.gradient_prob = gradient_prob

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Select elites
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.random() < self.crossover_prob:
                    p1, p2 = np.random.choice(elite_indices, 2, replace=False)
                    offspring = self.crossover(population[p1], population[p2])
                else:
                    offspring = population[np.random.choice(elite_indices)]

                # Apply quantum state update probabilistically
                if np.random.random() < self.quantum_prob:
                    offspring = self.quantum_state_update(offspring, best_individual)

                # Apply gradient boost probabilistically
                if np.random.random() < self.gradient_prob:
                    offspring = self.gradient_boost(offspring, func)

                # Mutate the offspring
                mutation_scale = self.adaptive_mutation_scale(evaluations)
                offspring += np.random.normal(0, mutation_scale, self.dimension)
                offspring = np.clip(offspring, -5, 5)

                new_population[i] = offspring

            # Evaluate the new population
            population = new_population
            fitness = np.array([func(ind) for ind in population])
            evaluations += self.population_size

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = population[current_best_idx]

        return best_fitness, best_individual

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def quantum_state_update(self, individual, best_individual):
        perturbation = np.random.normal(0, 0.1, self.dimension)
        return best_individual + perturbation * (best_individual - individual)

    def gradient_boost(self, individual, func):
        grad_est = np.zeros(self.dimension)
        fx = func(individual)
        h = 1e-5
        for i in range(self.dimension):
            x_new = individual.copy()
            x_new[i] += h
            grad_est[i] = (func(x_new) - fx) / h
        return individual - 0.01 * grad_est

    def adaptive_mutation_scale(self, evaluations):
        return self.mutation_intensity * np.exp(-0.05 * evaluations / self.budget)
