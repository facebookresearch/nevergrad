import numpy as np


class DualPhaseRefinedQuantumLocalSearchOptimizer:
    def __init__(
        self,
        budget,
        dim=5,
        population_size=60,
        elite_size=10,
        mutation_intensity=0.05,
        local_search_phase1=0.05,
        local_search_phase2=0.01,
    ):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.elite_size = elite_size
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.mutation_intensity = mutation_intensity
        self.local_search_phase1 = local_search_phase1
        self.local_search_phase2 = local_search_phase2

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(x) for x in population])

    def select_elites(self, population, fitnesses):
        elite_indices = np.argsort(fitnesses)[: self.elite_size]
        return population[elite_indices], fitnesses[elite_indices]

    def crossover(self, parent1, parent2):
        mask = np.random.rand(self.dim) < 0.5
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def mutate(self, individual):
        mutation = np.random.normal(0, self.mutation_intensity, self.dim)
        mutated = individual + mutation
        return np.clip(mutated, self.lower_bound, self.upper_bound)

    def local_search(self, func, candidate, intensity):
        for _ in range(10):  # perform 10 local search steps
            perturbation = np.random.uniform(-intensity, intensity, self.dim)
            new_candidate = candidate + perturbation
            new_candidate = np.clip(new_candidate, self.lower_bound, self.upper_bound)
            if func(new_candidate) < func(candidate):
                candidate = new_candidate
        return candidate

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(func, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            if evaluations < self.budget / 2:
                local_search_intensity = self.local_search_phase1
            else:
                local_search_intensity = self.local_search_phase2

            elites, elite_fitness = self.select_elites(population, fitness)
            new_population = elites.copy()  # start new population with elites
            for _ in range(self.population_size - self.elite_size):
                parents = np.random.choice(elites.shape[0], 2, replace=False)
                parent1, parent2 = elites[parents[0]], elites[parents[1]]
                offspring = self.crossover(parent1, parent2)
                offspring = self.mutate(offspring)
                offspring = self.local_search(
                    func, offspring, local_search_intensity
                )  # Perform local search on offspring
                new_population = np.vstack((new_population, offspring))

            new_fitness = self.evaluate_population(func, new_population)

            if np.min(new_fitness) < best_fitness:
                best_idx = np.argmin(new_fitness)
                best_individual = new_population[best_idx]
                best_fitness = new_fitness[best_idx]

            population = new_population
            fitness = new_fitness

            evaluations += self.population_size

        return best_fitness, best_individual
