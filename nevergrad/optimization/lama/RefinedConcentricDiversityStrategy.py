import numpy as np


class RefinedConcentricDiversityStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        islands=100,
        population_per_island=50,
        migration_interval=10,
        migration_rate=0.2,
        mutation_intensity=2.0,
        mutation_decay=0.95,
        elite_ratio=0.25,
        crossover_probability=0.9,
        tournament_size=3,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.islands = islands
        self.population_per_island = population_per_island
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.mutation_intensity = mutation_intensity
        self.mutation_decay = mutation_decay
        self.elite_ratio = elite_ratio
        self.elite_count = int(self.population_per_island * self.elite_ratio)
        self.crossover_probability = crossover_probability
        self.tournament_size = tournament_size
        self.total_population_size = self.islands * self.population_per_island

    def initialize_population(self):
        return np.random.uniform(
            self.lower_bound, self.upper_bound, (self.total_population_size, self.dimension)
        )

    def mutate(self, individual):
        mutation = np.random.normal(0, self.mutation_intensity, self.dimension)
        return np.clip(individual + mutation, self.lower_bound, self.upper_bound)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            alpha = np.random.rand(self.dimension)
            offspring = alpha * parent1 + (1 - alpha) * parent2
            return np.clip(offspring, self.lower_bound, self.upper_bound)
        else:
            return parent1.copy()

    def tournament_selection(self, population, fitness):
        indices = np.random.randint(0, population.shape[0], self.tournament_size)
        best_index = indices[np.argmin(fitness[indices])]
        return population[best_index]

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_count]
        return population[elite_indices]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        evaluations = self.total_population_size
        generation = 0

        while evaluations < self.budget:
            new_population = []
            for i in range(self.islands):
                start_idx = i * self.population_per_island
                end_idx = start_idx + self.population_per_island
                island_pop = population[start_idx:end_idx]
                island_fit = fitness[start_idx:end_idx]

                elites = self.select_elites(island_pop, island_fit)

                for _ in range(self.population_per_island - len(elites)):
                    parent1 = self.tournament_selection(island_pop, island_fit)
                    parent2 = self.tournament_selection(island_pop, island_fit)
                    child = self.crossover(parent1, parent2)
                    mutated_child = self.mutate(child)
                    new_population.append(mutated_child)

                new_population.extend(elites)

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

            if evaluations + self.total_population_size > self.budget:
                break

            if generation % self.migration_interval == 0:
                migrants = int(self.migration_rate * self.total_population_size)
                migration_indices = np.random.permutation(self.total_population_size)[:migrants]
                np.random.shuffle(migration_indices)
                population[migration_indices] = population[np.random.permutation(migration_indices)]

            current_best = population[np.argmin(fitness)]
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best

            evaluations += self.total_population_size
            generation += 1
            self.mutation_intensity *= self.mutation_decay

        return best_fitness, best_individual
