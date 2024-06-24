import numpy as np


class RefinedAdaptiveIslandEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        islands=5,
        population_per_island=20,
        migration_rate=0.1,
        mutation_intensity=0.5,
        mutation_decay=0.98,
        elite_ratio=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.islands = islands
        self.population_per_island = population_per_island
        self.migration_rate = migration_rate
        self.mutation_intensity = mutation_intensity
        self.mutation_decay = mutation_decay
        self.elite_ratio = elite_ratio
        self.elite_count = int(self.population_per_island * self.elite_ratio)
        self.total_population_size = self.islands * self.population_per_island

    def initialize_population(self):
        return np.random.uniform(
            self.lower_bound, self.upper_bound, (self.total_population_size, self.dimension)
        )

    def mutate(self, individual):
        mutation = np.random.normal(0, self.mutation_intensity, self.dimension)
        return np.clip(individual + mutation, self.lower_bound, self.upper_bound)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dimension)
        offspring = alpha * parent1 + (1 - alpha) * parent2
        return np.clip(offspring, self.lower_bound, self.upper_bound)

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_count]
        return population[elite_indices]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        evaluations = self.total_population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.islands):
                start_idx = i * self.population_per_island
                end_idx = start_idx + self.population_per_island
                island_pop = population[start_idx:end_idx]
                island_fit = fitness[start_idx:end_idx]

                elites = self.select_elites(island_pop, island_fit)

                # Fill the rest of the island population
                for _ in range(self.population_per_island - len(elites)):
                    parents = np.random.choice(island_pop.shape[0], 2, replace=False)
                    child = self.crossover(island_pop[parents[0]], island_pop[parents[1]])
                    mutated_child = self.mutate(child)
                    new_population.append(mutated_child)

                new_population.extend(elites)

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

            if evaluations + self.total_population_size > self.budget:
                break

            if np.random.rand() < self.migration_rate:
                np.random.shuffle(population)

            current_best = population[np.argmin(fitness)]
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best

            evaluations += self.total_population_size
            self.mutation_intensity *= self.mutation_decay

        return best_fitness, best_individual
