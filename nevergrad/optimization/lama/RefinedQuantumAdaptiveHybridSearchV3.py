import numpy as np


class RefinedQuantumAdaptiveHybridSearchV3:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=120,
        elite_frac=0.25,
        mutation_intensity=0.7,
        crossover_prob=0.75,
        quantum_prob=0.85,
        gradient_prob=0.65,
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
        # Initialize the population uniformly within the bounds.
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Select elites based on fitness
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population members
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.random() < self.crossover_prob:
                    # Perform crossover from two randomly selected elites
                    p1, p2 = np.random.choice(elite_indices, 2, replace=False)
                    offspring = self.crossover(population[p1], population[p2])
                else:
                    # Directly copy an elite
                    offspring = population[np.random.choice(elite_indices)]

                # Apply quantum state update with a probability
                if np.random.random() < self.quantum_prob:
                    offspring = self.quantum_state_update(offspring, best_individual)

                # Apply gradient enhancement with a probability
                if np.random.random() < self.gradient_prob:
                    offspring = self.gradient_boost(offspring, func)

                # Mutate the offspring
                mutation_scale = self.adaptive_mutation_scale(evaluations)
                offspring += np.random.normal(0, mutation_scale, self.dimension)
                offspring = np.clip(offspring, -5, 5)  # Ensure bounds are respected

                new_population[i] = offspring

            # Evaluate the new population
            fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Update best individual if improved
            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_fitness:
                best_fitness = fitness[current_best_idx]
                best_individual = new_population[current_best_idx]

            # Update the population
            population = new_population

        return best_fitness, best_individual

    def crossover(self, parent1, parent2):
        # Perform an alpha-blended crossover
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def quantum_state_update(self, individual, global_best):
        # Apply a quantum-inspired perturbation
        perturbation = np.random.normal(0, 0.1, self.dimension)
        return global_best + perturbation * (global_best - individual)

    def gradient_boost(self, individual, func):
        # Apply a simple numerical gradient approximation
        grad_est = np.zeros(self.dimension)
        fx = func(individual)
        h = 1e-5
        for i in range(self.dimension):
            x_new = individual.copy()
            x_new[i] += h
            grad_est[i] = (func(x_new) - fx) / h
        return individual - 0.01 * grad_est  # Update using a small learning rate

    def adaptive_mutation_scale(self, evaluations):
        # Reduce mutation scale as the number of evaluations increases
        return self.mutation_intensity * np.exp(-0.1 * evaluations / self.budget)
