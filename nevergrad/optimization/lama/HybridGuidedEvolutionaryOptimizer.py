import numpy as np


class HybridGuidedEvolutionaryOptimizer:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.mutation_factor = 0.8  # Mutation scaling factor
        self.crossover_rate = 0.7  # Probability of crossover
        self.adaptation_rate = 0.1  # Rate of adapting mutation factor
        self.elitism_rate = 0.1  # Proportion of elite individuals
        self.success_memory = []

    def adapt_mutation_factor(self):
        """Adapt the mutation factor based on moving average of recent successes"""
        if len(self.success_memory) > 10:
            success_rate = np.mean(self.success_memory[-10:])
            if success_rate > 0.2:
                self.mutation_factor *= 1 + self.adaptation_rate
            else:
                self.mutation_factor *= 1 - self.adaptation_rate
            self.mutation_factor = max(0.01, min(1.0, self.mutation_factor))  # Ensure within bounds
            self.crossover_rate = max(
                0.1, min(0.9, self.crossover_rate + (-0.05 if success_rate > 0.2 else 0.05))
            )

    def mutate(self, individual):
        """Apply mutation with dynamic adaptation"""
        mutation = np.random.normal(0, self.mutation_factor, self.dimension)
        mutant = individual + mutation
        return np.clip(mutant, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent1, parent2):
        """Blended crossover"""
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.uniform(-0.1, 1.1, self.dimension)
            return np.clip(alpha * parent1 + (1 - alpha) * parent2, self.bounds["lb"], self.bounds["ub"])
        return parent1  # No crossover occurred

    def __call__(self, func):
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        evaluations = len(population)

        while evaluations < self.budget:
            elite_size = int(self.elitism_rate * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]
            elite_individuals = population[elite_indices]

            offspring = []
            for _ in range(self.population_size - elite_size):
                parents = np.random.choice(self.population_size, 2, replace=False)
                child = self.crossover(population[parents[0]], population[parents[1]])
                mutated_child = self.mutate(child)
                offspring.append(mutated_child)

            offspring = np.vstack((elite_individuals, offspring))
            offspring_fitness = np.array([func(child) for child in offspring])
            evaluations += len(offspring)

            # Select new population
            combined = np.vstack((population, offspring))
            combined_fitness = np.concatenate((fitness, offspring_fitness))
            indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined[indices]
            fitness = combined_fitness[indices]

            # Update best solution
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]
                self.success_memory.append(1)
            else:
                self.success_memory.append(0)

            # Adapt parameters dynamically
            self.adapt_mutation_factor()

            if evaluations >= self.budget:
                break

        return best_fitness, best_individual
