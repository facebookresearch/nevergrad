import numpy as np


class ALDEEM:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.archive_size = 30
        self.mutation_scale = 0.5
        self.crossover_prob = 0.7
        self.elite_size = int(self.population_size * 0.2)
        self.memory = []

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, F):
        mutants = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(np.delete(np.arange(len(population)), best_idx), 3, replace=False)
            x1, x2, x3 = population[idxs]
            mutant_vector = np.clip(x1 + F * (x2 - x3), self.bounds[0], self.bounds[1])
            mutants[i] = mutant_vector
        return mutants

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.crossover_prob
        j_rand = np.random.randint(self.dimension)
        cross_points[j_rand] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            F = np.clip(np.random.normal(self.mutation_scale, 0.1), 0.1, 1.0)
            mutants = self.mutate(population, np.argmin(fitness), F)
            trials = np.array(
                [self.crossover(population[i], mutants[i]) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += self.population_size

            improvement_mask = fitness_trials < fitness
            population[improvement_mask] = trials[improvement_mask]
            fitness[improvement_mask] = fitness_trials[improvement_mask]

            # Maintain and utilize memory for diversity
            if evaluations % 500 == 0:
                self.memory.extend(population[np.argsort(fitness)[: self.elite_size]].tolist())
                if len(self.memory) > self.archive_size:
                    self.memory = self.memory[-self.archive_size :]
                reseed_indices = np.random.choice(self.population_size, size=self.elite_size, replace=False)
                population[reseed_indices] = np.array(self.memory[: self.elite_size])
                fitness[reseed_indices] = self.evaluate(population[reseed_indices], func)

            # Adaptive parameter control
            if evaluations % 100 == 0:
                self.mutation_scale *= 0.95 if np.min(fitness) < np.mean(fitness) else 1.05
                self.crossover_prob = max(
                    min(self.crossover_prob + (0.05 if np.min(fitness) < np.mean(fitness) else -0.05), 1.0),
                    0.1,
                )

        return np.min(fitness), population[np.argmin(fitness)]
