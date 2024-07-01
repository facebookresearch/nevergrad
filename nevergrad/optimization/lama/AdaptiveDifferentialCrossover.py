import numpy as np


class AdaptiveDifferentialCrossover:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 50
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def initialize_population(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, (self.pop_size, self.dim))

    def select_parents(self, population):
        idxs = np.random.choice(range(self.pop_size), 3, replace=False)
        return population[idxs]

    def mutate(self, parent1, parent2, parent3):
        return parent1 + self.mutation_factor * (parent2 - parent3)

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dim) < self.crossover_rate
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        bounds = func.bounds

        population = self.initialize_population(bounds)
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        for _ in range(self.budget // self.pop_size - 1):
            for i in range(self.pop_size):
                parent1, parent2, parent3 = self.select_parents(population)
                mutant = self.mutate(parent1, parent2, parent3)
                trial = self.crossover(population[i], mutant)

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            if np.min(fitness) < self.f_opt:
                self.f_opt = np.min(fitness)
                self.x_opt = population[np.argmin(fitness)]

        return self.f_opt, self.x_opt
