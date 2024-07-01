import numpy as np


class AdvancedBalancedAdaptiveElitistStrategyV2:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=50,
        elite_fraction=0.2,
        mutation_intensity=0.1,
        crossover_rate=0.7,
        recombination_factor=0.5,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity  # Initial mutation intensity factor
        self.crossover_rate = crossover_rate  # Probability of crossover
        self.recombination_factor = recombination_factor  # Weight factor for recombination

    def __call__(self, func):
        # Initialize the population within bounds [-5, 5]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Identify elite individuals
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if i < self.elite_count:
                    # Elite individuals are carried over unchanged
                    new_population[i] = population[elite_indices[i]]
                else:
                    # Generate new individuals by mutation and crossover
                    if np.random.random() < self.crossover_rate:
                        # Perform crossover
                        parent1, parent2 = elites[np.random.choice(len(elites), 2, replace=False)]
                        child = self.recombination(parent1, parent2)
                    else:
                        # Directly mutate an elite
                        parent = elites[np.random.randint(0, self.elite_count)]
                        child = self.mutate(parent, evaluations)

                    new_population[i] = np.clip(child, -5.0, 5.0)

                # Evaluate new individual's fitness
                new_fitness = func(new_population[i])
                if new_fitness < fitness[i]:
                    fitness[i] = new_fitness
                    population[i] = new_population[i]
                evaluations += 1

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_individual = new_population[i]

                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual

    def mutate(self, individual, evaluations):
        # Adaptive mutation scale decreases over time
        scale = self.mutation_intensity * (1 - (evaluations / self.budget))
        return individual + np.random.normal(0, scale, self.dimension)

    def recombination(self, parent1, parent2):
        # Blended recombination
        return self.recombination_factor * parent1 + (1 - self.recombination_factor) * parent2
