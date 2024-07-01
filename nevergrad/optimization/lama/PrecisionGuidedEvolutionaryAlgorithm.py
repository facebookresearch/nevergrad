import numpy as np


class PrecisionGuidedEvolutionaryAlgorithm:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.mutation_factor = 0.8  # Initial mutation scaling factor
        self.crossover_rate = 0.7  # Probability of crossover
        self.adaptation_rate = 0.1  # Rate at which mutation factor is adapted

    def adapt_mutation_factor(self, success_rate):
        """Adapt the mutation factor based on recent success rate"""
        if success_rate > 0.2:
            self.mutation_factor *= 1 + self.adaptation_rate
        else:
            self.mutation_factor *= 1 - self.adaptation_rate
        self.mutation_factor = max(0.01, min(1.0, self.mutation_factor))  # Ensure within bounds

    def mutate(self, individual):
        """Apply mutation with dynamic adaptation"""
        mutation = np.random.normal(0, self.mutation_factor, self.dimension)
        mutant = individual + mutation
        return np.clip(mutant, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent1, parent2):
        """Simulated binary crossover"""
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.uniform(-0.5, 1.5, self.dimension)
            return np.clip(alpha * parent1 + (1 - alpha) * parent2, self.bounds["lb"], self.bounds["ub"])
        return parent1  # No crossover occurred

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]

        successful_mutations = 0
        evaluations = len(population)

        # Evolutionary loop
        while evaluations < self.budget:
            offspring = []
            for individual in population:
                mutated = self.mutate(individual)
                f_mutated = func(mutated)
                evaluations += 1
                if f_mutated < func(individual):
                    offspring.append(mutated)
                    successful_mutations += 1
                else:
                    offspring.append(individual)

                if evaluations >= self.budget:
                    break

            offspring = np.array(
                [
                    self.crossover(offspring[i], offspring[np.random.randint(len(offspring))])
                    for i in range(len(offspring))
                ]
            )

            # Evaluate offspring
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += len(offspring)

            # Select new population
            combined = np.vstack((population, offspring))
            combined_fitness = np.concatenate((fitness, offspring_fitness))
            indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined[indices]
            fitness = combined_fitness[indices]

            # Update best solution
            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_individual = population[np.argmin(fitness)]

            # Adapt mutation factor based on success rate
            self.adapt_mutation_factor(successful_mutations / len(offspring))
            successful_mutations = 0  # Reset for the next generation

            if evaluations >= self.budget:
                break

        return best_fitness, best_individual
