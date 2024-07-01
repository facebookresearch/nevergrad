import numpy as np


class AdaptiveCrossoverElitistStrategyV6:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=120,
        elite_fraction=0.25,
        mutation_intensity=0.1,
        crossover_rate=0.85,
        adaptive_crossover=True,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.adaptive_crossover = adaptive_crossover

    def __call__(self, func):
        # Initialize the population within the bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
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
                if np.random.random() < self.crossover_rate:
                    # Perform adaptive crossover
                    parent1, parent2 = self.select_parents(elites, population)
                    child = self.recombine(parent1, parent2, evaluations)
                else:
                    # Mutation of an elite
                    parent = elites[np.random.choice(len(elites))]
                    child = self.mutate(parent, evaluations)

                new_population[i] = np.clip(child, -5.0, 5.0)
                new_fitness = func(new_population[i])
                evaluations += 1

                # Update the best solution found
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_individual = new_population[i]

                if evaluations >= self.budget:
                    break

            # Replace old population
            population = new_population
            fitness = np.array([func(individual) for individual in population])

        return best_fitness, best_individual

    def mutate(self, individual, evaluations):
        # Adaptive mutation intensity
        scale = self.mutation_intensity * np.exp(-evaluations / self.budget * 10)
        return individual + np.random.normal(0, scale, self.dimension)

    def recombine(self, parent1, parent2, evaluations):
        # Adaptive recombination based on the stage of optimization
        alpha = np.random.uniform(0.4, 0.6)
        if self.adaptive_crossover:
            alpha *= np.exp(-evaluations / self.budget)
        return alpha * parent1 + (1 - alpha) * parent2

    def select_parents(self, elites, population):
        if self.adaptive_crossover:
            parent1 = elites[np.random.choice(len(elites))]
            parent2 = population[np.random.randint(0, self.population_size)]
        else:
            parent_indices = np.random.choice(len(elites), 2, replace=False)
            parent1, parent2 = elites[parent_indices[0]], elites[parent_indices[1]]
        return parent1, parent2
