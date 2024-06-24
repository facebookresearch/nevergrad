import numpy as np


class AdvancedSpatialAdaptiveConvergenceOptimizer:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=120,
        initial_step_size=1.5,
        step_decay=0.95,
        elite_ratio=0.05,
        mutation_intensity=0.05,
        local_search_prob=0.3,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.step_size = initial_step_size
        self.step_decay = step_decay
        self.elite_count = int(population_size * elite_ratio)
        self.mutation_intensity = mutation_intensity
        self.local_search_prob = local_search_prob

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual, scale):
        mutation = np.random.normal(0, scale * self.mutation_intensity, self.dimension)
        return np.clip(individual + mutation, self.bounds[0], self.bounds[1])

    def local_search(self, individual):
        tweaks = np.random.normal(0, self.step_size * 0.02, self.dimension)  # More refined local search
        return np.clip(individual + tweaks, self.bounds[0], self.bounds[1])

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(func, population)
        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            scale = self.step_size * (self.step_decay**generation)
            new_population = np.array(
                [self.mutate(population[i], scale) for i in range(self.population_size)]
            )
            new_fitness = self.evaluate_population(func, new_population)

            if np.random.rand() < self.local_search_prob:
                for idx in range(self.population_size):
                    candidate = self.local_search(new_population[idx])
                    candidate_fitness = func(candidate)
                    if candidate_fitness < new_fitness[idx]:
                        new_population[idx] = candidate
                        new_fitness[idx] = candidate_fitness

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            elite_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[elite_indices]
            fitness = combined_fitness[elite_indices]

            current_best = population[np.argmin(fitness)]
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best

            evaluations += self.population_size
            generation += 1

        return best_fitness, best_individual
